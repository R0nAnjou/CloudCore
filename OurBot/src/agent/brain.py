"""主决策编排: 每回合把状态分派给 白天经济 / 夜战 / 任务 三大模块。

阶段0 建造计划: 武器站点 + 围墙圈(预留入口) — 沿用 Demo 的塔防布局思路。
"""
import logging
from typing import Any

from . import defense, economy, protocol as P, tasks
from .grid import next_step
from .memory import Memory
from .protocol import (
    Pos,
    STATION,
    Unit,
    WALL,
    WEAPON_BUILD_COST,
    buy_command,
    distance,
    move_command,
    station_footprint,
    use_command,
)

LOGGER = logging.getLogger(__name__)

MEMORY = Memory()


def decide(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    turn = P.Turn.load(payload)
    MEMORY.rounds_seen += 1

    # 1) 记录新闻(推理类/长上下文类任务的输入)
    day = P.day_index(turn.round_no)
    MEMORY.remember_news(day, turn.world_official_news, turn.world_folk_legends)

    # 2) 反馈学习: 上回合动作结果
    _learn_from_feedback(turn)

    # 3) 召唤宝藏结果纠错
    if turn.last_summon_result:
        tasks.handle_summon_result(turn, MEMORY)

    # 4) 分派
    commands: dict[int, dict[str, Any]] = {}
    prompt = ""
    execute_cmd = ""
    if turn.is_day:
        prompt, execute_cmd = _day_phase(turn, commands)
    else:
        defense.night(turn, set(), commands)

    MEMORY.last_commands = {str(k): v for k, v in commands.items()}
    return {
        "roleCommandMap": {str(k): v for k, v in commands.items()},
        "prompt": prompt,
        "executeCmd": execute_cmd,
    }


def _learn_from_feedback(turn: P.Turn) -> None:
    # 动作失败 -> 若是 build 记为非法位置
    failed: dict[int, bool] = {}
    for role_id, ok in turn.last_action_results.items():
        if not ok:
            failed[role_id] = False
    if not failed:
        return
    last = MEMORY.last_commands
    for role_id, _ in failed.items():
        cmd = last.get(str(role_id))
        if not cmd:
            continue
        if cmd.get("action") == "build" and cmd.get("targetPos"):
            pos_raw = cmd["targetPos"][0]
            P.Pos  # noqa: B018  (类型引用)
            MEMORY.note_build_result(
                Pos(int(pos_raw["x"]), int(pos_raw["y"])), str(cmd.get("name")), False
            )
            LOGGER.warning("learned invalid build site %s", pos_raw)


def _day_phase(
    turn: P.Turn,
    commands: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    prompt = ""
    execute_cmd = ""
    plan = _build_plan(turn)
    claimed: set[Pos] = set()
    claimed_sites: set[Pos] = set()
    roles = turn.controllable()
    workers = turn.workers()
    pioneer_roles = turn.pioneers()

    # 工人: 建造/卖矿/采集 + 顺手修墙
    for worker in workers:
        if economy.use_repair_if_needed(turn, worker, commands):
            continue
        economy.worker_day(turn, worker, plan, claimed, claimed_sites, MEMORY, commands)

    # 购物: 派第一个工人去武器商店买清单商品(若背包有空间)
    _dispatch_shopping(turn, workers, claimed, commands)

    # 开拓者: 任务引擎
    if pioneer_roles:
        prompt, execute_cmd = tasks.pioneer(turn, pioneer_roles[0], MEMORY, claimed, commands)

    return prompt, execute_cmd


def _dispatch_shopping(
    turn: P.Turn,
    workers: tuple[Unit, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    want = economy.shopping_list(turn)
    if not want:
        return
    shop = turn.weapon_shop_pos()
    if shop is None:
        return
    for name, num in want:
        # 已有人背这个物品且量足够 -> 跳过购买, 直接由持有者去用
        holder = next((w for w in workers if w.count(name) >= num), None)
        if holder is not None:
            _carry_to_use(turn, holder, name, claimed, commands)
            return
        buyer = next((w for w in workers if not w.backpack_full), None)
        if buyer is None:
            return
        if distance(buyer.pos, shop) <= 1:
            commands[buyer.unit_id] = buy_command(name, num)
        else:
            step = next_step(turn, buyer, shop)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[buyer.unit_id] = move_command(step)
        return


def _carry_to_use(
    turn: P.Turn,
    holder: Unit,
    name: str,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    station = turn.station()
    if station is None:
        return
    footprint = station_footprint(station.pos)
    target = footprint[0]  # 基地左上格
    if name.startswith("Station"):
        if distance(holder.pos, target) <= 1:
            commands[holder.unit_id] = use_command(name, target)
        else:
            step = next_step(turn, holder, target)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[holder.unit_id] = move_command(step)
    elif name == P.WALL_FIXER:
        if economy.use_repair_if_needed(turn, holder, commands):
            return
        for wall in turn.walls():
            if wall.health < 1000 and distance(holder.pos, wall.pos) <= 1:
                commands[holder.unit_id] = use_command(name, wall.pos)
                return


# ---- 建造计划 ----------------------------------------------------------
def _build_plan(turn: P.Turn) -> economy.Plan:
    station = turn.station()
    if station is None:
        return economy.Plan()
    footprint = station_footprint(station.pos)

    # 武器站点: 基地足印外圈 3 个空地(面对地图中央一侧优先)
    center = Pos(turn.width // 2, turn.height // 2)
    sites = sorted(
        (
            pos
            for pos in _ring(footprint, radius=1)
            if turn.land(pos) and MEMORY.build_allowed(pos, P.GATLING)
        ),
        key=lambda p: (distance(p, center), p.x, p.y),
    )
    tower_sites = tuple(sites[:3])
    tower_kinds = economy.TOWER_LOADOUT[: len(tower_sites)]

    # 围墙圈: 半径 2, 预留朝地图中央的入口
    entrance_dir = _dominant_axis(footprint[0], center)
    wall_sites_list = [
        pos
        for pos in _ring(footprint, radius=2)
        if turn.land(pos) and MEMORY.build_allowed(pos, WALL)
    ]
    # 预留入口: 最靠近中央的 2 格不建墙
    wall_sites_list.sort(key=lambda p: (distance(p, center), p.x, p.y))
    entrance = set(wall_sites_list[:2]) if entrance_dir else set()
    wall_sites = tuple(p for p in wall_sites_list if p not in entrance)
    return economy.Plan(tower_sites=tower_sites, tower_kinds=tower_kinds, wall_sites=wall_sites)


def _ring(footprint: tuple[Pos, ...], radius: int) -> list[Pos]:
    xs = [p.x for p in footprint]
    ys = [p.y for p in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cells: list[Pos] = []
    for x in range(xmin - radius, xmax + radius + 1):
        for y in range(ymin - radius, ymax + radius + 1):
            if xmin - radius < x < xmax + radius and ymin - radius < y < ymax + radius:
                continue  # 只留环边
            cells.append(Pos(x, y))
    return cells


def _dominant_axis(a: Pos, b: Pos) -> bool:
    return abs(a.x - b.x) + abs(a.y - b.y) > 0

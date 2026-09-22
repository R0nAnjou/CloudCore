"""主决策编排: 每回合把状态分派给 白天经济 / 夜战 / 任务 三大模块。

阶段0 建造计划: 武器站点 + 围墙圈(预留入口) — 沿用 Demo 的塔防布局思路。
"""
import logging
from typing import Any

from . import defense, economy, protocol as P, tasks
from .grid import next_step_adjacent, next_step_adjacent_to_any
from .memory import Memory
from .protocol import (
    Pos,
    Unit,
    WALL,
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
    if turn.last_summon_result or MEMORY.treasure_pending_round:
        tasks.handle_summon_result(turn, MEMORY)

    # 4) 分派
    commands: dict[int, dict[str, Any]] = {}
    prompt = ""
    execute_cmd = ""
    if turn.is_day:
        prompt, execute_cmd = _day_phase(turn, commands)
    else:
        prompt, execute_cmd = _night_phase(turn, commands)

    _log_defense_snapshot(turn, commands)

    MEMORY.last_commands = {str(k): v for k, v in commands.items()}
    return {
        "roleCommandMap": {str(k): v for k, v in commands.items()},
        "prompt": prompt,
        "executeCmd": execute_cmd,
    }


def _night_phase(
    turn: P.Turn,
    commands: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    prompt = ""
    execute_cmd = ""
    claimed: set[Pos] = set()
    protected: set[int] = set()
    pioneers = turn.pioneers()
    if pioneers:
        pioneer = pioneers[0]
        prompt, execute_cmd = tasks.pioneer(
            turn, pioneer, MEMORY, claimed, commands, allow_new_task=False
        )
        if (
            MEMORY.task_state != "idle"
            or pioneer.unit_id in commands
            or (
                MEMORY.treasure_pos is not None
                and MEMORY.treasure_items
                and not MEMORY.treasure_done
            )
        ):
            protected.add(pioneer.unit_id)
    # 修复包能在夜晚使用；仅抢救身旁濒危的墙，远处工人仍优先操作炮台。
    for worker in turn.workers():
        if economy.move_or_repair_wall(
            turn, worker, claimed, commands, urgent_only=True,
        ):
            protected.add(worker.unit_id)
    defense.night(turn, claimed, commands, protected)
    return prompt, execute_cmd


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
        LOGGER.warning("round %d action failed role=%d action=%s", turn.round_no,
                       role_id, cmd.get("action"))
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
    workers = turn.workers()
    pioneer_roles = turn.pioneers()
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1

    # 已持有的修复包优先处理，哪怕购买后金币已花光。
    for worker in workers:
        economy.move_or_repair_wall(
            turn, worker, claimed, commands, urgent_only=day_round >= 60,
        )

    if day_round >= 60:
        # 夜幕将至：工人必须回到墙圈内，否则封门后无法操控炮台。
        _pre_night_fortify(turn, workers, plan, claimed, claimed_sites, commands)
    else:
        # 购物先于普通采建调度，避免覆盖采矿/建造动作。
        _dispatch_shopping(turn, workers, claimed, commands)
        for worker in workers:
            if worker.unit_id not in commands:
                economy.worker_day(turn, worker, plan, claimed, claimed_sites, MEMORY, commands)

    # 开拓者: 任务引擎
    if pioneer_roles:
        prompt, execute_cmd = tasks.pioneer(turn, pioneer_roles[0], MEMORY, claimed, commands)

    return prompt, execute_cmd


def _inside_defense(turn: P.Turn, role: Unit) -> bool:
    station = turn.station()
    return station is not None and min(
        distance(role.pos, pos) for pos in station_footprint(station.pos)
    ) <= 1


def _pre_night_fortify(
    turn: P.Turn,
    workers: tuple[Unit, ...],
    plan: economy.Plan,
    claimed: set[Pos],
    claimed_sites: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    station = turn.station()
    if station is None:
        return
    footprint = station_footprint(station.pos)
    standing = turn.occupied_cells()
    pending_sites = [
        pos for pos in plan.wall_sites if pos not in standing and pos not in claimed_sites
    ]
    for worker in workers:
        if worker.unit_id in commands:
            continue
        if not _inside_defense(turn, worker):
            step = next_step_adjacent_to_any(turn, worker, list(footprint))
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[worker.unit_id] = move_command(step)
            continue
        # 只建身边的墙，不再离开内圈去采矿或追逐远处建造点。
        if worker.count(P.STONE) == 0:
            continue
        site = next(
            (pos for pos in pending_sites
             if pos not in claimed_sites and distance(worker.pos, pos) <= 1),
            None,
        )
        if site is not None:
            claimed_sites.add(site)
            commands[worker.unit_id] = P.build_command(site, WALL)
            continue
        # 沿内圈靠近尚未封住的墙位；不允许为了建墙再次走出防线。
        blocked = turn.blocked(worker)
        inner_steps = [
            pos for pos in _ring(footprint, radius=1)
            if distance(worker.pos, pos) == 1 and turn.land(pos)
            and pos not in blocked and pos not in claimed
        ]
        for target in pending_sites:
            if target in claimed_sites:
                continue
            closer = [
                pos for pos in inner_steps
                if distance(pos, target) < distance(worker.pos, target)
            ]
            if closer:
                step = min(closer, key=lambda pos: (distance(pos, target), pos.x, pos.y))
                claimed.add(step)
                commands[worker.unit_id] = move_command(step)
                break


def _dispatch_shopping(
    turn: P.Turn,
    workers: tuple[Unit, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    # 已购升级券必须优先兑现；购买后金币下降，不能再依赖 shopping_list 才发现它。
    carried_upgrades = (
        P.STATION_UP_V1, P.STATION_UP_V2, P.WEAPON_UP_V1, P.WEAPON_UP_V2,
        P.WALL_UP_V1, P.WALL_UP_V2,
    )
    for name in carried_upgrades:
        holder = next(
            (worker for worker in workers
             if worker.has(name) and worker.unit_id not in commands
             and _upgrade_has_target(turn, name)),
            None,
        )
        if holder is not None:
            _carry_to_use(turn, holder, name, claimed, commands)
            return

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
            if holder.unit_id in commands:
                return
            _carry_to_use(turn, holder, name, claimed, commands)
            return
        reserved = _reserved_gold(turn, commands)
        price = turn.shop_prices.get(name, 0) * num
        if turn.gold - reserved < price:
            return
        buyer = next(
            (
                w for w in workers
                if not w.backpack_full
                and w.unit_id not in commands
            ),
            None,
        )
        if buyer is None:
            return
        if distance(buyer.pos, shop) <= 1:
            commands[buyer.unit_id] = buy_command(name, num)
        else:
            step = next_step_adjacent(turn, buyer, shop)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[buyer.unit_id] = move_command(step)
        return


def _upgrade_has_target(turn: P.Turn, name: str) -> bool:
    if name in {P.STATION_UP_V1, P.STATION_UP_V2}:
        station = turn.station()
        level = 1 if name == P.STATION_UP_V1 else 2
        return station is not None and station.level == level
    if name in {P.WEAPON_UP_V1, P.WEAPON_UP_V2}:
        level = 1 if name == P.WEAPON_UP_V1 else 2
        return any(weapon.level == level for weapon in turn.weapons())
    if name in {P.WALL_UP_V1, P.WALL_UP_V2}:
        level = 1 if name == P.WALL_UP_V1 else 2
        return any(wall.level == level for wall in turn.walls())
    return False


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
        if min(distance(holder.pos, pos) for pos in footprint) <= 1:
            commands[holder.unit_id] = use_command(name, target)
        else:
            step = next_step_adjacent_to_any(turn, holder, list(footprint))
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[holder.unit_id] = move_command(step)
    elif name.startswith("Weapon"):
        wanted_level = 1 if name == P.WEAPON_UP_V1 else 2
        weapon = next((w for w in turn.weapons() if w.level == wanted_level), None)
        if weapon is None:
            return
        if distance(holder.pos, weapon.pos) <= 1:
            commands[holder.unit_id] = use_command(name, weapon.pos)
        else:
            step = next_step_adjacent(turn, holder, weapon.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[holder.unit_id] = move_command(step)
    elif name.startswith("WallUpgrade"):
        wanted_level = 1 if name == P.WALL_UP_V1 else 2
        wall = next((w for w in turn.walls() if w.level == wanted_level), None)
        if wall is None:
            return
        if distance(holder.pos, wall.pos) <= 1:
            commands[holder.unit_id] = use_command(name, wall.pos)
        else:
            step = next_step_adjacent(turn, holder, wall.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[holder.unit_id] = move_command(step)
    elif name == P.WALL_FIXER:
        if economy.use_repair_if_needed(turn, holder, commands):
            return
        wall = min(turn.walls(), key=lambda w: w.health, default=None)
        if wall is not None:
            step = next_step_adjacent(turn, holder, wall.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[holder.unit_id] = move_command(step)


def _reserved_gold(turn: P.Turn, commands: dict[int, dict[str, Any]]) -> int:
    total = 0
    for cmd in commands.values():
        if cmd.get("action") == "build" and cmd.get("name") in P.TOWER_TYPES:
            total += P.WEAPON_BUILD_COST
        elif cmd.get("action") == "buy":
            total += turn.shop_prices.get(str(cmd.get("name") or ""), 0) * int(cmd.get("num") or 1)
    return total


# ---- 建造计划 ----------------------------------------------------------
def _build_plan(turn: P.Turn) -> economy.Plan:
    station = turn.station()
    if station is None:
        return economy.Plan()
    footprint = station_footprint(station.pos)

    # 武器站点: 基地足印外圈 3 个空地(面对地图中央一侧优先)
    center = Pos(turn.width // 2, turn.height // 2)
    occupied = turn.occupied_cells()
    sites = sorted(
        (
            pos
            for pos in _ring(footprint, radius=1)
            if turn.land(pos) and pos not in occupied
        ),
        key=lambda p: (distance(p, center), p.x, p.y),
    )
    chosen_sites: list[Pos] = []
    chosen_kinds: list[str] = []
    existing_kinds = {weapon.kind for weapon in turn.weapons()}
    missing_kinds = [kind for kind in economy.TOWER_LOADOUT if kind not in existing_kinds]
    needed = max(0, 3 - len(turn.weapons()))
    for kind in missing_kinds[:needed]:
        site = next(
            (p for p in sites if p not in chosen_sites and MEMORY.build_allowed(p, kind)),
            None,
        )
        if site is not None:
            chosen_sites.append(site)
            chosen_kinds.append(kind)
    tower_sites = tuple(chosen_sites)
    tower_kinds = tuple(chosen_kinds)

    # 围墙圈: 半径 2。背向中央暂留一格通道，工人回到内圈后再封门。
    wall_sites_list = [
        pos
        for pos in _ring(footprint, radius=2)
        if turn.land(pos) and MEMORY.build_allowed(pos, WALL)
    ]
    # 朝敌人方向的墙先建，首夜即使材料不足也先挡住正面。
    wall_sites_list.sort(key=lambda p: (distance(p, center), p.x, p.y))
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    if wall_sites_list and not (
        day_round >= 60 and all(_inside_defense(turn, worker) for worker in turn.workers())
    ):
        wall_sites_list.pop()  # 最远离中央的一格作为临时入口
    wall_sites = tuple(wall_sites_list)
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


def _log_defense_snapshot(turn: P.Turn, commands: dict[int, dict[str, Any]]) -> None:
    station = turn.station()
    if turn.round_no in {1, 50, 60} or 70 <= turn.round_no <= 80 or turn.round_no % 10 == 0:
        LOGGER.info(
            "defense round=%d station_hp=%s walls=%d towers=%d workers=%d "
            "gold=%d stone=%d robots=%d attacks=%d",
            turn.round_no, station.health if station else 0, len(turn.walls()),
            len(turn.weapons()), len(turn.workers()), turn.gold,
            sum(worker.count(P.STONE) for worker in turn.workers()), len(turn.robots),
            sum(command.get("action") == "attack" for command in commands.values()),
        )
    for role_id, command in commands.items():
        if command.get("action") == "build" or (
            command.get("action") == "use" and command.get("name") == P.WALL_FIXER
        ):
            LOGGER.info("defense round=%d role=%d action=%s name=%s target=%s",
                        turn.round_no, role_id, command.get("action"),
                        command.get("name"), command.get("targetPos"))

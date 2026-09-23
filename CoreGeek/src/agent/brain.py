"""主决策编排: 每回合把状态分派给 白天经济 / 夜战 / 任务 三大模块。

阶段0 建造计划: 武器站点 + 围墙圈(预留入口) — 沿用 Demo 的塔防布局思路。
"""
import copy
import logging
import threading
from itertools import combinations, permutations
from typing import Any, Callable

from . import defense, economy, protocol as P, tasks
from .grid import adjacent_stands, neighbours, path_to_any, next_step_to_any, next_step_adjacent, next_step_adjacent_to_any
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
_DECISION_LOCK = threading.Lock()
STRATEGY_VERSION = "survival-loop-20260923f"


def decide(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    global MEMORY
    turn = P.Turn.load(payload)
    if turn.round_no == 1 and MEMORY.current_round > 1:
        MEMORY = Memory()
    MEMORY.current_round = turn.round_no
    MEMORY.note_positions(turn.controllable())
    if turn.round_no == 1:
        LOGGER.info("strategy_version=%s", STRATEGY_VERSION)
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

    _resolve_command_conflicts(turn, commands)
    _log_defense_snapshot(turn, commands)

    MEMORY.last_commands = {str(k): v for k, v in commands.items()}
    return {
        "roleCommandMap": {str(k): v for k, v in commands.items()},
        "prompt": prompt,
        "executeCmd": execute_cmd,
    }


def decide_transactional(
    payload: dict[str, Any],
    publish: Callable[[dict[str, dict[str, Any]]], bool],
) -> bool:
    """在内存副本上决策；只有服务器仍等待结果时才提交跨回合状态。"""
    global MEMORY
    with _DECISION_LOCK:
        live_memory = MEMORY
        MEMORY = copy.deepcopy(live_memory)
        try:
            result = decide(payload)
            candidate_memory = MEMORY
        except Exception:
            MEMORY = live_memory
            raise
        MEMORY = live_memory
        if not publish(result):
            return False
        MEMORY = candidate_memory
        return True


def _night_phase(
    turn: P.Turn,
    commands: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    prompt = ""
    execute_cmd = ""
    claimed: set[Pos] = set()
    protected: set[int] = set()
    pioneers = turn.pioneers()
    night_miner = _night_miner(turn)
    post_wave_safe = _post_wave_safe(turn)
    if pioneers:
        pioneer = pioneers[0]
        prompt, execute_cmd = tasks.pioneer(
            turn, pioneer, MEMORY, claimed, commands, allow_new_task=False,
            allow_treasure=False, departure_round=turn.round_no,
        )
        if (
            pioneer.unit_id in commands
        ):
            protected.add(pioneer.unit_id)
    for role in turn.controllable():
        if role.unit_id in commands:
            continue
        max_hp = 220 if role.kind == P.WORKER else 200
        if role.has(P.MEDICINE) and role.health < max_hp * 0.45:
            commands[role.unit_id] = use_command(P.MEDICINE)
            protected.add(role.unit_id)
    station = turn.station()
    if station and station.level < 3 and station.health < 1500 * station.level * 0.65:
        item = P.STATION_UP_V1 if station.level == 1 else P.STATION_UP_V2
        for worker in turn.workers():
            if worker.unit_id not in commands and worker.has(item) and _inside_defense(turn, worker):
                _carry_to_use(turn, worker, item, claimed, commands)
                protected.add(worker.unit_id)
                break
    # A 工留在基地附近维护。浪潮中只修身边的危墙，清场后再走动补齐全部残墙。
    repairers = tuple(
        worker for worker in turn.workers()
        if night_miner is None or worker.unit_id != night_miner.unit_id
    )
    for worker in repairers:
        if worker.unit_id not in commands and economy.move_or_repair_wall(
            turn, worker, claimed, commands, urgent_only=not post_wave_safe, memory=MEMORY,
        ):
            protected.add(worker.unit_id)
    if night_miner is not None and night_miner.unit_id not in commands:
        economy.worker_night_safe(turn, night_miner, MEMORY, claimed, commands)
        protected.add(night_miner.unit_id)
    # 清场后 A 工先补完墙，再拆后门恢复采矿；B 工从始至终在安全矿区生产。
    if post_wave_safe:
        walls_healthy = _walls_healthy(turn)
        if walls_healthy:
            _open_gate(turn, claimed, commands, force=True)
            # 墙修好后再兑现升级；没攒够钱就继续采矿。
            _dispatch_shopping(turn, repairers, claimed, commands, allow_returning=True)
        else:
            # 没有修复包时，A 工先去买包，不能因为手上无包就直接转去采矿。
            _dispatch_shopping(
                turn, repairers, claimed, commands,
                allow_returning=True, prioritize_repair=True,
            )
        for worker in turn.workers():
            if not walls_healthy and worker in repairers:
                continue
            if worker.unit_id not in commands:
                economy.worker_night_safe(turn, worker, MEMORY, claimed, commands)
    # 开拓者在共享炮位轮射三炮。若开拓者本回合被任务收尾占用，仅 A 工兜底。
    if pioneers and pioneers[0].unit_id not in commands:
        protected.update(worker.unit_id for worker in turn.workers())
    defense.night(turn, claimed, commands, protected)
    return prompt, execute_cmd


def _post_wave_safe(turn: P.Turn) -> bool:
    """本方浪潮清空即进入夜间修墙/生产/反击阶段。"""
    station = turn.station()
    return (station is not None and station.health > 0
            and not defense.robots_threatening_us(turn))


def _walls_healthy(turn: P.Turn) -> bool:
    return all(
        wall.health >= economy._wall_max_health(wall.level) * economy.HEALTHY_WALL_RATIO
        for wall in turn.walls()
    )


def _learn_from_feedback(turn: P.Turn) -> None:
    # 动作失败 -> 建造点退避；移动点短期拉黑，避免原地反复撞同一格。
    failed: dict[int, bool] = {}
    for role_id, ok in turn.last_action_results.items():
        cmd = MEMORY.last_commands.get(str(role_id), {})
        if cmd.get("action") in {"buy", "sell", "use", "build", "remove"}:
            LOGGER.info("action_result round=%d role=%d action=%s name=%s num=%s success=%s",
                        turn.round_no, role_id, cmd.get("action"), cmd.get("name"), cmd.get("num"), ok)
        if ok and cmd.get("action") == "build" and cmd.get("targetPos"):
            MEMORY.note_build_result(Pos.load(cmd["targetPos"][0]), str(cmd.get("name")), True)
        if cmd.get("action") == "move" and cmd.get("targetPos"):
            MEMORY.note_move_result(role_id, Pos.load(cmd["targetPos"][0]), ok)
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
            LOGGER.warning("build site temporarily deferred %s", pos_raw)


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
    night_miner = _night_miner(turn)
    pioneer_roles = turn.pioneers()
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    MEMORY.current_round = turn.round_no
    if day_round == 1:
        MEMORY.returning_roles.clear()

    carried_upgrade_names = {
        P.STATION_UP_V1, P.STATION_UP_V2, P.WEAPON_UP_V1, P.WEAPON_UP_V2,
        P.WALL_UP_V1, P.WALL_UP_V2,
    }
    has_carried_upgrade = any(
        any(worker.has(name) for name in carried_upgrade_names) for worker in workers
    )
    # 基地濒危、危墙缺修复包、角色濒死时，不受 15 面墙成长门槛限制。
    emergency_want = economy.emergency_shopping_list(turn)
    if emergency_want:
        _dispatch_shopping(
            turn, workers, claimed, commands,
            allow_returning=True, wanted=emergency_want, carried_first=False,
        )
    # 非紧急的已购升级券随后兑现，B 若持券也不能在夜幕前被先送出基地。
    if has_carried_upgrade:
        _dispatch_shopping(turn, workers, claimed, commands, allow_returning=True)

    # 有固定出入口的完整昼夜循环：只有防线和封门石头都就绪，才送 B 出城。
    release_night_miner = (
        night_miner is not None
        and _night_miner_release_ready(turn, plan, night_miner)
    )
    night_miner_exiting = (
        release_night_miner and night_miner is not None
        and _inside_defense(turn, night_miner)
    )
    _open_gate(
        turn, claimed, commands, allow_night_miner_exit=release_night_miner,
    )
    if day_round >= 55 and release_night_miner:
        _send_night_miner_outside(turn, night_miner, claimed, commands)
    _clear_gate_lane(turn, claimed, commands)
    for role in turn.controllable():
        if (release_night_miner and night_miner is not None
                and role.unit_id == night_miner.unit_id):
            MEMORY.returning_roles.discard(role.unit_id)
            continue
        travel = _home_distance(turn, role)
        if (night_miner_exiting and role.kind == P.WORKER
                and night_miner is not None and role.unit_id != night_miner.unit_id
                and not _inside_defense(turn, role)):
            MEMORY.returning_roles.discard(role.unit_id)
            continue
        if day_round >= 60 or day_round + travel + 3 >= 70:
            MEMORY.returning_roles.add(role.unit_id)

    # 已持有的修复包优先处理，哪怕购买后金币已花光。
    for worker in workers:
        if worker.unit_id not in commands and worker.has(P.MEDICINE) and worker.health < 132:
            commands[worker.unit_id] = use_command(P.MEDICINE)
        if (worker.unit_id not in commands
                and (night_miner is None or worker.unit_id != night_miner.unit_id)):
            economy.move_or_repair_wall(
                turn, worker, claimed, commands,
                urgent_only=worker.unit_id in MEMORY.returning_roles,
                memory=MEMORY,
            )
        if (worker.unit_id in MEMORY.returning_roles and worker.unit_id not in commands
                and not _inside_defense(turn, worker)):
            _return_home(turn, worker, claimed, commands)

    if day_round >= 60:
        # 夜幕将至：A 工回墙内维护，B 工继续外部现金流，开拓者回共享炮位。
        if (release_night_miner and night_miner is not None
                and night_miner.unit_id not in commands):
            economy.worker_day(
                turn, night_miner, plan, claimed, claimed_sites, MEMORY, commands,
                income_only=True,
            )
        defenders = tuple(
            worker for worker in workers
            if (not release_night_miner or night_miner is None
                or worker.unit_id != night_miner.unit_id)
            and not (night_miner_exiting and not _inside_defense(turn, worker))
        )
        _pre_night_fortify(turn, defenders, plan, claimed, claimed_sites, commands)
    else:
        built_wall_positions = {wall.pos for wall in turn.walls()}
        built_walls = sum(pos in built_wall_positions for pos in plan.wall_sites)
        front_complete = all(pos in built_wall_positions for pos in _front_wall_sites(turn))
        first_night_quota_met = built_walls >= 15 and front_complete
        # 第一优先级是三炮与至少 3/4 围墙；达到生存线后才让购物占用工人回合。
        if first_night_quota_met and not has_carried_upgrade:
            _dispatch_shopping(turn, workers, claimed, commands)
        for index, worker in enumerate(workers):
            if worker.unit_id not in commands and worker.unit_id not in MEMORY.returning_roles:
                gate = _gate_position(turn)
                economy.worker_day(
                    turn, worker, plan, claimed, claimed_sites, MEMORY, commands,
                    income_only=(len(workers) > 1 and index == len(workers) - 1
                                 and first_night_quota_met),
                    force_stone=(index == 0 and day_round >= 45 and gate is not None
                                 and not any(wall.pos == gate for wall in turn.walls())),
                )

    # 开拓者: 任务引擎
    if pioneer_roles:
        pioneer = pioneer_roles[0]
        departure = turn.round_no + max(0, 70 - day_round - _home_distance(turn, pioneer) - 3)
        task_active = (MEMORY.task_state in tasks.ACTIVE_TASK_STATES
                       or bool(turn.phase_task))
        recovering = _recover_pioneer(
            turn, pioneer, claimed, commands, allow_travel=not task_active,
        )
        if not recovering:
            prompt, execute_cmd = tasks.pioneer(
                turn, pioneer, MEMORY, claimed, commands,
                allow_new_task=day_round < 50 and pioneer.unit_id not in MEMORY.returning_roles,
                allow_treasure=pioneer.unit_id not in MEMORY.returning_roles,
                departure_round=departure,
            )
        if (pioneer.unit_id in MEMORY.returning_roles and MEMORY.task_state == "idle"
                and pioneer.unit_id not in commands):
            _return_home(turn, pioneer, claimed, commands)

    if day_round >= 66:
        staging_roles = [
            worker for worker in workers
            if night_miner is None or worker.unit_id != night_miner.unit_id
        ]
        if pioneer_roles and MEMORY.task_state == "idle":
            staging_roles.append(pioneer_roles[0])
        _stage_gunners(turn, tuple(staging_roles), claimed, commands)

    return prompt, execute_cmd


def _recover_pioneer(
    turn: P.Turn,
    pioneer: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow_travel: bool = True,
) -> bool:
    """开拓者是三炮唯一轮射手；低血量时白天先自救，不能带 5 点血进下一夜。"""
    if pioneer.unit_id in commands or pioneer.health >= 150:
        return False
    if pioneer.has(P.MEDICINE):
        commands[pioneer.unit_id] = use_command(P.MEDICINE)
        return True
    if not allow_travel:
        return False
    shop = turn.weapon_shop_pos()
    price = turn.shop_prices.get(P.MEDICINE, 10)
    if shop is None or pioneer.backpack_full or turn.gold - _reserved_gold(turn, commands) < price:
        return False
    if distance(pioneer.pos, shop) <= 1:
        commands[pioneer.unit_id] = buy_command(P.MEDICINE, 1)
        return True
    step = next_step_adjacent(
        turn, pioneer, shop, reserved=_move_reserved(pioneer, claimed),
    )
    if step is None or step in claimed:
        return False
    claimed.add(step)
    commands[pioneer.unit_id] = move_command(step)
    return True


def _inside_defense(turn: P.Turn, role: Unit) -> bool:
    station = turn.station()
    return station is not None and min(
        distance(role.pos, pos) for pos in station_footprint(station.pos)
    ) <= 1


def _night_miner(turn: P.Turn) -> Unit | None:
    """两名工人中编号较大的 B 工负责昼夜连续采矿；单工局仍必须回防。"""
    workers = turn.workers()
    return workers[-1] if len(workers) >= 2 else None


def _night_defenders_inside(turn: P.Turn) -> bool:
    """封门只等待开拓者和 A 工，刻意忽略留在墙外的 B 工。"""
    miner = _night_miner(turn)
    return all(
        _inside_defense(turn, role)
        for role in turn.controllable()
        if miner is None or role.unit_id != miner.unit_id
    )


def _night_defenders_ready(turn: P.Turn) -> bool:
    """三炮成型后必须等开拓者站稳共享炮位，才允许把唯一入口封死。"""
    if not _night_defenders_inside(turn):
        return False
    pioneers = turn.pioneers()
    if len(turn.weapons()) >= 3 and pioneers and MEMORY.gunner_pos is not None:
        return pioneers[0].pos == MEMORY.gunner_pos
    return True


def _home_distance(turn: P.Turn, role: Unit) -> int:
    station = turn.station()
    if station is None or _inside_defense(turn, role):
        return 0
    path = path_to_any(turn, role, adjacent_stands(turn, role, station_footprint(station.pos)))
    # 不可达时提前开始恢复路线，而不是等最后几回合才发现。
    return len(path) if path is not None else 20


def _return_home(turn: P.Turn, role: Unit, claimed: set[Pos], commands: dict) -> None:
    station = turn.station()
    if station is None or _inside_defense(turn, role):
        return
    step = next_step_adjacent_to_any(
        turn, role, list(station_footprint(station.pos)), reserved=_move_reserved(role, claimed),
    )
    if step is not None:
        claimed.add(step)
        commands[role.unit_id] = move_command(step)


def _send_night_miner_outside(
    turn: P.Turn, worker: Unit, claimed: set[Pos], commands: dict[int, dict[str, Any]],
) -> None:
    """夜幕前把 B 工明确送出后门，避免占住开拓者的三炮共享站位。"""
    if worker.unit_id in commands:
        return
    station = turn.station()
    gate = _gate_position(turn)
    if station is None or gate is None:
        return
    footprint = station_footprint(station.pos)
    if min(distance(worker.pos, pos) for pos in footprint) > 2:
        return
    outside = {
        pos for pos in neighbours(gate)
        if turn.land(pos) and min(distance(pos, cell) for cell in footprint) > 2
    }
    step = next_step_to_any(
        turn, worker, outside, reserved=_move_reserved(worker, claimed),
    )
    if step is not None and step not in claimed:
        claimed.add(step)
        commands[worker.unit_id] = move_command(step)
        return
    # 紧凑三炮会形成单格通道。若 A 正堵在门内，先让 A 临时走到门外，
    # B 通过后 A 再按 returning_roles 返回封门，避免两人原地互相等待。
    blockers = [
        role for role in turn.workers()
        if role.unit_id != worker.unit_id and role.unit_id not in commands
        and distance(role.pos, gate) <= 1
    ]
    for blocker in blockers:
        blocker_step = next_step_to_any(
            turn, blocker, outside, reserved=_move_reserved(blocker, claimed),
        )
        if blocker_step is not None and blocker_step not in claimed:
            claimed.add(blocker_step)
            commands[blocker.unit_id] = move_command(blocker_step)
            return


def _night_miner_release_ready(
    turn: P.Turn, plan: economy.Plan, night_miner: Unit,
) -> bool:
    """防线未达到首夜生存线时，B 工必须留下补墙并在夜前返城。"""
    day = P.day_index(turn.round_no)
    if MEMORY.night_miner_committed_day.get(night_miner.unit_id) == day:
        return True
    if len(turn.weapons()) < 3:
        return False
    wall_positions = {wall.pos for wall in turn.walls()}
    planned_wall_positions = set(plan.wall_sites)
    built_walls = len(wall_positions & planned_wall_positions)
    if built_walls < 15 or not set(_front_wall_sites(turn)) <= wall_positions:
        return False
    pioneers = turn.pioneers()
    if not pioneers or pioneers[0].health < 100 or MEMORY.gunner_pos is None:
        return False
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    if distance(pioneers[0].pos, MEMORY.gunner_pos) > max(0, 66 - day_round):
        return False
    repairers = [worker for worker in turn.workers()
                 if worker.unit_id != night_miner.unit_id]
    ready = bool(repairers and any(worker.has(P.STONE) for worker in repairers))
    if ready:
        MEMORY.night_miner_committed_day[night_miner.unit_id] = day
    return ready


def _gate_position(turn: P.Turn) -> Pos | None:
    station = turn.station()
    if station is None:
        return None
    # 任务地图的黄色围墙建造区就是基地足印外半径 2；日志已验证这些格可建。
    ring = [pos for pos in _ring(station_footprint(station.pos), 2) if turn.land(pos)]
    if MEMORY.gate_pos not in ring:
        # 门固定在背敌面。此前按地图中心距离选门，在部分出生点会把缺口留在正面。
        MEMORY.gate_pos = min(
            ring, key=lambda pos: (_enemy_projection(turn, pos), pos.x, pos.y), default=None,
        )
    return MEMORY.gate_pos


def _enemy_anchor(turn: P.Turn) -> tuple[float, float]:
    enemy_station = next((role for role in turn.enemy_roles if role.kind == P.STATION), None)
    if enemy_station is not None:
        cells = station_footprint(enemy_station.pos)
        return (sum(pos.x for pos in cells) / len(cells),
                sum(pos.y for pos in cells) / len(cells))
    if turn.enemy_roles:
        return (sum(role.pos.x for role in turn.enemy_roles) / len(turn.enemy_roles),
                sum(role.pos.y for role in turn.enemy_roles) / len(turn.enemy_roles))
    return float(turn.width // 2), float(turn.height // 2)


def _base_center(turn: P.Turn) -> tuple[float, float]:
    station = turn.station()
    if station is None:
        return 0.0, 0.0
    cells = station_footprint(station.pos)
    return (sum(pos.x for pos in cells) / len(cells),
            sum(pos.y for pos in cells) / len(cells))


def _enemy_projection(turn: P.Turn, pos: Pos) -> float:
    """越大越靠近敌方，越小越适合作为后门。"""
    base_x, base_y = _base_center(turn)
    enemy_x, enemy_y = _enemy_anchor(turn)
    return (pos.x - base_x) * (enemy_x - base_x) + (pos.y - base_y) * (enemy_y - base_y)


def _front_wall_sites(turn: P.Turn) -> tuple[Pos, ...]:
    station = turn.station()
    if station is None:
        return ()
    ring = [
        pos for pos in _ring(station_footprint(station.pos), 2)
        if turn.land(pos) and MEMORY.build_allowed(pos, WALL)
    ]
    ring.sort(key=lambda pos: (-_enemy_projection(turn, pos), pos.x, pos.y))
    # 20 格围墙圈取最朝敌的 1/3 作为不可欠账的正面墙。
    return tuple(ring[:max(6, len(ring) // 3)])


def _move_reserved(role: Unit, claimed: set[Pos]) -> set[Pos]:
    return set(claimed) | MEMORY.failed_move_cells(role.unit_id)


def _open_gate(
    turn: P.Turn,
    claimed: set[Pos],
    commands: dict,
    *,
    force: bool = False,
    allow_night_miner_exit: bool = True,
) -> None:
    gate = _gate_position(turn)
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    if gate is None or (not force and day_round >= 60 and _night_defenders_ready(turn)):
        return
    if not any(wall.pos == gate for wall in turn.walls()):
        return
    if not force and day_round >= 55 and not allow_night_miner_exit:
        # 防线未达标时不为 B 主动拆门；只有墙外防守角色确实需要返城才开。
        miner = _night_miner(turn)
        outsiders = any(
            not _inside_defense(turn, role)
            for role in turn.controllable()
            if miner is None or role.unit_id != miner.unit_id
        )
        if not outsiders:
            return
    miner = _night_miner(turn)
    workers = [
        worker for worker in turn.workers()
        if worker.unit_id not in commands
        and (miner is None or worker.unit_id != miner.unit_id)
    ]
    for worker in sorted(workers, key=lambda role: distance(role.pos, gate)):
        if distance(worker.pos, gate) <= 1:
            commands[worker.unit_id] = P.remove_command(gate)
            LOGGER.info("gate round=%d action=open pos=%s", turn.round_no, gate)
            return
        step = next_step_adjacent(turn, worker, gate, reserved=_move_reserved(worker, claimed))
        if step is not None:
            claimed.add(step)
            commands[worker.unit_id] = move_command(step)
            return


def _clear_gate_lane(turn: P.Turn, claimed: set[Pos], commands: dict) -> None:
    gate = _gate_position(turn)
    station = turn.station()
    if gate is None or station is None:
        return
    miner = _night_miner(turn)
    outsiders = any(
        not _inside_defense(turn, role)
        for role in turn.controllable()
        if miner is None or role.unit_id != miner.unit_id
    )
    closing = ((turn.round_no - 1) % P.ROUNDS_PER_DAY + 1 >= 60
               and not any(w.pos == gate for w in turn.walls()))
    if not outsiders and not closing:
        return
    inner = _ring(station_footprint(station.pos), 1)
    for role in turn.controllable():
        if (role.unit_id in commands or not _inside_defense(turn, role)
                or distance(role.pos, gate) > 1
                or role.kind == P.PIONEER and MEMORY.task_state != "idle"):
            continue
        if miner is not None and role.unit_id == miner.unit_id:
            continue  # B 正沿单格通道出城，不能再被让路逻辑推回基地内。
        if role.kind == P.PIONEER and role.pos == MEMORY.gunner_pos:
            continue
        miner_on_gunner = (
            miner is not None and role.unit_id == miner.unit_id
            and role.pos == MEMORY.gunner_pos and bool(turn.pioneers())
        )
        if (not outsiders and role.kind == P.WORKER and role.has(P.STONE)
                and not miner_on_gunner):
            continue  # 持有石头的工人留在门边封门，其他角色让出施工位。
        candidates = [pos for pos in inner if distance(role.pos, pos) == 1
                      and distance(pos, gate) > 1 and turn.land(pos)
                      and pos not in turn.blocked(role) and pos not in claimed]
        if candidates:
            step = min(candidates, key=lambda pos: (distance(pos, role.pos), pos.x, pos.y))
            commands[role.unit_id] = move_command(step)
            claimed.add(step)


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
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    footprint = station_footprint(station.pos)
    standing = turn.occupied_cells()
    pending_sites = [
        pos for pos in plan.wall_sites if pos not in standing and pos not in claimed_sites
    ]
    for worker in workers:
        if worker.unit_id in commands:
            continue
        if not _inside_defense(turn, worker):
            step = next_step_adjacent_to_any(
                turn, worker, list(footprint), reserved=_move_reserved(worker, claimed),
            )
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[worker.unit_id] = move_command(step)
            continue
        gate = _gate_position(turn)
        if (gate in pending_sites and gate not in claimed_sites and worker.has(P.STONE)
                and _night_defenders_ready(turn)):
            if distance(worker.pos, gate) <= 1:
                commands[worker.unit_id] = P.build_command(gate, WALL)
                claimed_sites.add(gate)
                claimed.add(gate)
            else:
                # 半径 2 的门从墙内侧半径 1 施工，封门后 A 工仍留在防线内。
                gate_stands = {
                    pos for pos in _ring(footprint, 1)
                    if turn.land(pos) and pos not in turn.blocked(worker)
                    and distance(pos, gate) <= 1
                }
                step = next_step_to_any(
                    turn, worker, gate_stands, reserved=_move_reserved(worker, claimed),
                )
                if step is not None:
                    commands[worker.unit_id] = move_command(step)
                    claimed.add(step)
            continue
        if day_round >= 66:
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
            claimed.add(site)
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
def _stage_gunners(
    turn: P.Turn,
    roles: tuple[Unit, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """优先把开拓者送到三炮共享站位；开拓者阵亡时再由工人分炮值守。"""
    available_roles = [role for role in roles if role.unit_id not in commands]
    pioneer = next((role for role in available_roles if role.kind == P.PIONEER), None)
    shared = _shared_gunner_position(turn, pioneer) if pioneer is not None else None
    if pioneer is not None and shared is not None:
        if pioneer.pos != shared:
            step = next_step_to_any(
                turn, pioneer, {shared}, reserved=_move_reserved(pioneer, claimed),
            )
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[pioneer.unit_id] = move_command(step)
        return
    towers = sorted(
        turn.weapons(),
        key=lambda tower: ({P.ROCKET: 0, P.RAILGUN: 1, P.GATLING: 2}.get(tower.kind, 3),
                           tower.unit_id),
    )[:len(roles)]
    if not towers:
        return
    if not available_roles:
        return
    selected_towers = towers[:len(available_roles)]
    assignment = min(
        permutations(available_roles, len(selected_towers)),
        key=lambda candidates: sum(
            distance(role.pos, tower.pos)
            for tower, role in zip(selected_towers, candidates)
        ),
    )
    for tower, worker in zip(selected_towers, assignment):
        if distance(worker.pos, tower.pos) <= 1:
            continue
        step = next_step_adjacent(
            turn, worker, tower.pos, reserved=_move_reserved(worker, claimed),
        )
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[worker.unit_id] = move_command(step)


def _shared_gunner_position(turn: P.Turn, moving: Unit | None = None) -> Pos | None:
    towers = turn.weapons()
    if not towers:
        return MEMORY.gunner_pos
    blocked = turn.blocked(moving) if moving is not None else turn.occupied_cells()
    candidates = {
        pos for pos in neighbours(towers[0].pos)
        if turn.land(pos) and (moving is not None and pos == moving.pos or pos not in blocked)
        and all(distance(pos, tower.pos) <= 1 for tower in towers)
    }
    if MEMORY.gunner_pos in candidates:
        return MEMORY.gunner_pos
    station = turn.station()
    if station is None or not candidates:
        return None
    return min(candidates, key=lambda pos: (
        min(distance(pos, cell) for cell in station_footprint(station.pos)), pos.x, pos.y,
    ))


def _dispatch_shopping(
    turn: P.Turn,
    workers: tuple[Unit, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow_returning: bool = False,
    prioritize_repair: bool = False,
    wanted: list[tuple[str, int]] | None = None,
    carried_first: bool = True,
) -> None:
    # 已购升级券必须优先兑现；购买后金币下降，不能再依赖 shopping_list 才发现它。
    carried_upgrades = (
        P.STATION_UP_V1, P.STATION_UP_V2, P.WEAPON_UP_V1, P.WEAPON_UP_V2,
        P.WALL_UP_V1, P.WALL_UP_V2,
    )
    if carried_first:
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

    want = list(wanted) if wanted is not None else economy.shopping_list(turn)
    if (prioritize_repair and turn.walls()
            and not any(worker.has(P.WALL_FIXER) for worker in workers)):
        want = [(P.WALL_FIXER, 1), *[item for item in want if item[0] != P.WALL_FIXER]]
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
        maintenance = {
            P.WALL_FIXER, P.STATION_UP_V1, P.STATION_UP_V2,
            P.WEAPON_UP_V1, P.WEAPON_UP_V2, P.WALL_UP_V1, P.WALL_UP_V2,
        }
        buyers = (sorted(workers, key=lambda role: role.health) if name == P.MEDICINE
                  else list(workers) if name in maintenance else list(reversed(workers)))
        buyer = next(
            (
                w for w in buyers
                if not w.backpack_full
                and w.unit_id not in commands
                and (allow_returning or w.unit_id not in MEMORY.returning_roles)
            ),
            None,
        )
        if buyer is None:
            return
        if distance(buyer.pos, shop) <= 1:
            commands[buyer.unit_id] = buy_command(name, num)
        else:
            step = next_step_adjacent(
                turn, buyer, shop, reserved=_move_reserved(buyer, claimed),
            )
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
            step = next_step_adjacent_to_any(
                turn, holder, list(footprint), reserved=_move_reserved(holder, claimed),
            )
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[holder.unit_id] = move_command(step)
    elif name.startswith("Weapon"):
        wanted_level = 1 if name == P.WEAPON_UP_V1 else 2
        weapon = next((w for w in sorted(
            turn.weapons(),
            key=lambda unit: {P.ROCKET: 0, P.RAILGUN: 1, P.GATLING: 2}.get(unit.kind, 3),
        ) if w.level == wanted_level), None)
        if weapon is None:
            return
        if distance(holder.pos, weapon.pos) <= 1:
            commands[holder.unit_id] = use_command(name, weapon.pos)
        else:
            step = next_step_adjacent(
                turn, holder, weapon.pos, reserved=_move_reserved(holder, claimed),
            )
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
            step = next_step_adjacent(
                turn, holder, wall.pos, reserved=_move_reserved(holder, claimed),
            )
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[holder.unit_id] = move_command(step)
    elif name == P.WALL_FIXER:
        if economy.use_repair_if_needed(turn, holder, commands):
            return
        wall = min(turn.walls(), key=lambda w: w.health, default=None)
        if wall is not None:
            step = next_step_adjacent(
                turn, holder, wall.pos, reserved=_move_reserved(holder, claimed),
            )
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[holder.unit_id] = move_command(step)
    elif name == P.MEDICINE:
        commands[holder.unit_id] = use_command(name)


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

    # 武器只能建在任务地图蓝区：日志验证为基地足印外半径 1。
    occupied = turn.occupied_cells() - {role.pos for role in turn.controllable()}
    sites = sorted(
        (
            pos
            for pos in _ring(footprint, radius=1)
            if turn.land(pos) and pos not in occupied
        ),
        key=lambda p: (-_enemy_projection(turn, p), p.x, p.y),
    )
    chosen_sites: list[Pos] = []
    chosen_kinds: list[str] = []
    needed = max(0, 3 - len(turn.weapons()))
    existing = {tower.pos for tower in turn.weapons()}
    cached = [pos for pos in MEMORY.tower_layout if pos not in existing]
    if needed and (len(cached) != needed or any(
        pos in occupied or not turn.land(pos) or not MEMORY.build_allowed(pos, P.ROCKET)
        for pos in cached
    )):
        candidates = [pos for pos in sites if MEMORY.build_allowed(pos, P.ROCKET)]
        layouts: list[
            tuple[tuple[float, float, float, int], tuple[Pos, ...], Pos, Pos]
        ] = []
        inner = {pos for pos in _ring(footprint, 1) if turn.land(pos) and pos not in occupied}
        wall_ring = {
            pos for pos in _ring(footprint, 2)
            if turn.land(pos) and MEMORY.build_allowed(pos, WALL)
        }
        protected_front = set(sorted(
            wall_ring,
            key=lambda pos: (-_enemy_projection(turn, pos), pos.x, pos.y),
        )[:max(6, len(wall_ring) // 3)])
        for chosen in combinations(candidates, needed):
            towers = existing | set(chosen)
            stands = inner - set(chosen)
            if len(towers) != 3 or not stands:
                continue
            shared_cells = [
                pos for pos in stands
                if all(distance(pos, tower) <= 1 for tower in towers)
            ]
            for shared in shared_cells:
                # 单共享炮位会被基地和三座炮围成口袋，因此入口仍须与炮位相邻；
                # 但绝不能占用最朝敌的正面六格。布局评分先看炮台覆盖，再选侧后门。
                gates = list(wall_ring)
                usable_gates = [
                    gate for gate in gates
                    if gate not in protected_front
                    and distance(gate, shared) <= 1
                    and any(
                        distance(inner_pos, gate) <= 1
                        for inner_pos in stands - {shared}
                    )
                ]
                if not usable_gates:
                    continue
                gate = min(
                    usable_gates,
                    key=lambda pos: (_enemy_projection(turn, pos), pos.x, pos.y),
                )
                projections = [_enemy_projection(turn, pos) for pos in towers]
                score = (
                    min(projections),
                    sum(projections),
                    -_enemy_projection(turn, gate),
                    -distance(shared, gate),
                )
                layouts.append((score, chosen, shared, gate))
        if layouts:
            _, selected, shared, gate = max(layouts, key=lambda item: item[0])
            cached = list(selected)
            MEMORY.tower_layout = (
                tuple(sorted(existing, key=lambda pos: (pos.x, pos.y))) + tuple(cached)
            )
            MEMORY.gunner_pos = shared
            MEMORY.gate_pos = gate
        else:
            cached = candidates[:needed]
            MEMORY.tower_layout = tuple(existing) + tuple(cached)
    sites = cached if needed else sites
    for kind in economy.TOWER_LOADOUT[:needed]:
        site = next(
            (p for p in sites if p not in chosen_sites and MEMORY.build_allowed(p, kind)),
            None,
        )
        if site is not None:
            chosen_sites.append(site)
            chosen_kinds.append(kind)
    tower_sites = tuple(chosen_sites)
    tower_kinds = tuple(chosen_kinds)

    # 围墙只能建在任务地图黄区：日志验证为基地足印外半径 2。
    wall_sites_list = [
        pos
        for pos in _ring(footprint, radius=2)
        if turn.land(pos) and MEMORY.build_allowed(pos, WALL)
    ]
    # 严格按敌方方向排序，首夜先完成正面，背敌面的门最后封。
    wall_sites_list.sort(key=lambda p: (-_enemy_projection(turn, p), p.x, p.y))
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    gate = _gate_position(turn)
    if not (day_round >= 60 and _night_defenders_ready(turn)):
        wall_sites_list = [pos for pos in wall_sites_list if pos != gate]
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
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    if day_round in {1, 50, 60} or 70 <= day_round <= 80 or turn.round_no % 10 == 0:
        LOGGER.info(
            "defense round=%d station_hp=%s walls=%d towers=%d workers=%d "
            "gold=%d stone=%d robots=%d ready_towers=%d attacks=%d station_level=%d score=%d",
            turn.round_no, station.health if station else 0, len(turn.walls()),
            len(turn.weapons()), len(turn.workers()), turn.gold,
            sum(worker.count(P.STONE) for worker in turn.workers()), len(turn.robots),
            sum(any(distance(role.pos, tower.pos) <= 1
                    for role in turn.controllable()) for tower in turn.weapons()),
            sum(command.get("action") == "attack" for command in commands.values()),
            station.level if station else 0, turn.total_score,
        )
        LOGGER.info("roles round=%d state=%s", turn.round_no, [
            (role.unit_id, role.pos.x, role.pos.y, role.health,
             "return" if role.unit_id in MEMORY.returning_roles
             else MEMORY.task_state if role.kind == P.PIONEER
             else MEMORY.worker_modes.get(role.unit_id, "idle"),
             commands.get(role.unit_id, {}).get("action", "hold")) for role in turn.controllable()
        ])
        for tower in turn.weapons():
            adjacent = [role.unit_id for role in turn.controllable() if distance(role.pos, tower.pos) <= 1]
            in_range = sum(distance(robot.pos, tower.pos) <= tower.range_of_attack() for robot in turn.robots)
            command = commands.get(tower.unit_id, {})
            reason = ("fire" if command.get("action") == "attack" else "day" if turn.is_day
                      else "cooldown" if tower.cooldown else "no_target" if not in_range
                      else "no_gunner" if not adjacent else "gunner_busy_or_covered")
            LOGGER.info("tower round=%d id=%d level=%d cooldown=%d adjacent=%s in_range=%d action=%s controller=%s",
                        turn.round_no, tower.unit_id, tower.level, tower.cooldown, adjacent,
                        in_range, reason, command.get("controllerId", ""))
    for role_id, command in commands.items():
        if command.get("action") in {"build", "use", "buy", "sell", "remove"}:
            LOGGER.info("defense round=%d role=%d action=%s name=%s target=%s",
                        turn.round_no, role_id, command.get("action"),
                        command.get("name"), command.get("targetPos"))


def _resolve_command_conflicts(turn: P.Turn, commands: dict[int, dict[str, Any]]) -> None:
    """最后检查跨模块的移动争格、换位、建筑占位及重复控制角色。"""
    controllers: set[int] = set()
    destinations: set[Pos] = set()
    role_positions = {role.pos for role in turn.controllable()}
    for key, command in list(commands.items()):
        if command.get("action") == "attack":
            role_id = int(command["controllerId"])
            if role_id in commands or role_id in controllers:
                commands.pop(key)
            else:
                controllers.add(role_id)
        elif command.get("action") == "move":
            pos = Pos.load(command["targetPos"][0])
            role_id = int(key)
            if MEMORY.would_oscillate(role_id, pos):
                LOGGER.warning("oscillation stopped round=%d role=%d target=%s",
                               turn.round_no, role_id, pos)
                MEMORY.note_move_result(role_id, pos, False)
                commands.pop(key)
            elif pos in destinations or pos in role_positions:
                commands.pop(key)
            else:
                destinations.add(pos)
    for key, command in list(commands.items()):
        if command.get("action") == "build" and Pos.load(command["targetPos"][0]) in destinations:
            commands.pop(key)

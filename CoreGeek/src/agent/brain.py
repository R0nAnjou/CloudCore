"""主决策编排: 每回合把状态分派给 白天经济 / 夜战 / 任务 三大模块。

阶段0 建造计划: 武器站点 + 围墙圈(预留入口) — 沿用 Demo 的塔防布局思路。
"""
import copy
import logging
import threading
from itertools import permutations
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
STRATEGY_VERSION = "central-wall-l3-two-operator-20260924"
UPGRADE_ITEMS = (
    P.STATION_UP_V1, P.STATION_UP_V2, P.WEAPON_UP_V1, P.WEAPON_UP_V2,
    P.WALL_UP_V1, P.WALL_UP_V2,
)


def decide(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    global MEMORY
    turn = P.Turn.load(payload)
    _reset_match_if_needed(turn)
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


def _reset_match_if_needed(turn: P.Turn) -> bool:
    """回合号回退、阵营切换或基地坐标突变都视为新对局。"""
    global MEMORY
    station = turn.station()
    station_pos = station.pos if station is not None else None
    changed = bool(MEMORY.current_round) and (
        turn.round_no < MEMORY.current_round
        or (MEMORY.match_team is not None and MEMORY.match_team != turn.team_type)
        or (MEMORY.match_station_pos is not None
            and station_pos is not None
            and MEMORY.match_station_pos != station_pos)
    )
    if changed:
        MEMORY = Memory()
    MEMORY.match_team = turn.team_type
    MEMORY.match_station_pos = station_pos
    return changed


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
    workers = turn.workers()
    night_miner = _night_miner(turn)
    builder = _builder_operator(turn)
    post_wave_safe = _post_wave_safe(turn)
    threats = defense.robots_threatening_us(turn)

    # 进攻阶段不允许任务或购物抢占两个炮手。若白天任务仍未结束，立即结束任务状态，
    # 保证开拓者可参与轮射；正常情况下白天的 departure_round 会更早完成返防。
    if not post_wave_safe and pioneers and MEMORY.task_state in tasks.ACTIVE_TASK_STATES:
        tasks.pioneer(
            turn, pioneers[0], MEMORY, claimed, commands,
            allow_new_task=False, allow_treasure=False, departure_round=turn.round_no,
        )

    # 清场后才释放开拓者做任务。剩余夜间 + 下一整个白天都可用于任务，但必须在
    # 下一晚前返回固定炮位。这正是对战日志中 round112 开始继续任务的时间窗。
    if post_wave_safe and pioneers:
        pioneer = pioneers[0]
        day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
        next_night = turn.round_no + (P.ROUNDS_PER_DAY - day_round) + P.DAY_ROUNDS + 1
        departure = next_night - _home_distance(turn, pioneer) - 3
        available = max(0, departure - turn.round_no)
        prompt, execute_cmd = tasks.pioneer(
            turn, pioneer, MEMORY, claimed, commands,
            allow_new_task=True, allow_treasure=True,
            departure_round=departure, available_rounds=available,
        )
        if pioneer.unit_id in commands:
            protected.add(pioneer.unit_id)
    for role in turn.controllable():
        if role.unit_id in commands:
            continue
        max_hp = 220 if role.kind == P.WORKER else 200
        if role.has(P.MEDICINE) and role.health < max_hp * 0.45:
            commands[role.unit_id] = use_command(P.MEDICINE)
            protected.add(role.unit_id)
    # 浪潮出现前可把已经买好的升级券立即送达，避免带着券进入整晚；一旦见过
    # 本晚机器人，必须等连续清场确认后才允许离开炮位。
    day = P.day_index(turn.round_no)
    if (not post_wave_safe and not threats and day not in MEMORY.wave_seen_days
            and builder is not None and builder.unit_id not in commands
            and any(builder.has(name) for name in UPGRADE_ITEMS)):
        _dispatch_shopping(
            turn, (builder,), claimed, commands, allow_returning=True,
        )
        if builder.unit_id in commands:
            protected.add(builder.unit_id)
    station = turn.station()
    if station and station.level < 3 and station.health < 1500 * station.level * 0.50:
        item = P.STATION_UP_V1 if station.level == 1 else P.STATION_UP_V2
        for worker in turn.workers():
            if worker.unit_id not in commands and worker.has(item) and _inside_defense(turn, worker):
                _carry_to_use(turn, worker, item, claimed, commands)
                protected.add(worker.unit_id)
                break
    repairers = tuple(worker for worker in workers if builder is not None
                      and worker.unit_id == builder.unit_id)
    if night_miner is not None and night_miner.unit_id not in commands:
        economy.worker_night_safe(turn, night_miner, MEMORY, claimed, commands)
        protected.add(night_miner.unit_id)

    # 清场后的调度顺序：危墙保命 > 已购升级券/成长采购 > 普通维修 > 继续生产。
    # 进攻尚未清空时不提前占用 A 工，它必须与开拓者共同轮射三炮；冷却回合的
    # 身边维修由 defense.night 统一安排。
    if post_wave_safe:
        for worker in repairers:
            if worker.unit_id not in commands and economy.move_or_repair_wall(
                turn, worker, claimed, commands, urgent_only=True, memory=MEMORY,
            ):
                protected.add(worker.unit_id)
        if builder is not None and builder.unit_id not in commands:
            _dispatch_shopping(
                turn, repairers, claimed, commands,
                allow_returning=True, prioritize_repair=False,
            )
            if builder.unit_id in commands:
                protected.add(builder.unit_id)
        for worker in repairers:
            if worker.unit_id not in commands and economy.move_or_repair_wall(
                turn, worker, claimed, commands, urgent_only=False, memory=MEMORY,
            ):
                protected.add(worker.unit_id)
        for worker in repairers:
            if worker.unit_id not in commands:
                economy.worker_night_safe(turn, worker, MEMORY, claimed, commands)
                protected.add(worker.unit_id)

    # 战斗阶段两名固定炮手轮射三炮；清场后仅未被任务/生产占用的角色参与反击。
    defense.night(
        turn, claimed, commands, protected,
        allow_counterfire=post_wave_safe,
    )
    return prompt, execute_cmd


def _post_wave_safe(turn: P.Turn) -> bool:
    """见过本方浪潮后连续两回合无威胁，才进入夜间生产阶段。"""
    station = turn.station()
    if station is None or station.health <= 0:
        return False
    day = P.day_index(turn.round_no)
    threats = defense.robots_threatening_us(turn)
    if threats:
        MEMORY.wave_seen_days.add(day)
        MEMORY.wave_clear_streak[day] = 0
        MEMORY.wave_clear_observed_round[day] = turn.round_no
        return False
    if day not in MEMORY.wave_seen_days:
        return False
    if MEMORY.wave_clear_observed_round.get(day) != turn.round_no:
        MEMORY.wave_clear_streak[day] = MEMORY.wave_clear_streak.get(day, 0) + 1
        MEMORY.wave_clear_observed_round[day] = turn.round_no
    return MEMORY.wave_clear_streak.get(day, 0) >= 2


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
    builder = _builder_operator(turn)
    pioneer_roles = turn.pioneers()
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    MEMORY.current_round = turn.round_no
    if day_round == 1:
        MEMORY.returning_roles.clear()

    has_carried_upgrade = any(
        any(worker.has(name) for name in UPGRADE_ITEMS) for worker in workers
    )
    # 基地濒危、危墙缺修复包、角色濒死时不受成长阶段限制。
    emergency_want = economy.emergency_shopping_list(turn)
    if emergency_want:
        _dispatch_shopping(
            turn, workers, claimed, commands,
            allow_returning=True, wanted=emergency_want, carried_first=False,
        )
    # 非紧急的已购升级券随后兑现。购物默认由建造工承担，矿工不因另一名工人
    # 持券而停止生产。
    if has_carried_upgrade:
        _dispatch_shopping(turn, workers, claimed, commands, allow_returning=True)

    # C 形防线背面永久开放，矿工从第 1 回合起就可以出城，绝不再拆墙开门。
    release_night_miner = night_miner is not None
    for role in turn.controllable():
        if (release_night_miner and night_miner is not None
                and role.unit_id == night_miner.unit_id):
            MEMORY.returning_roles.discard(role.unit_id)
            continue
        travel = _home_distance(turn, role)
        if day_round >= 60 or day_round + travel + 3 >= 70:
            MEMORY.returning_roles.add(role.unit_id)

    # 已持有的修复包优先处理，哪怕购买后金币已花光。
    for worker in workers:
        if worker.unit_id not in commands and worker.has(P.MEDICINE) and worker.health < 132:
            commands[worker.unit_id] = use_command(P.MEDICINE)
        if (worker.unit_id not in commands and builder is not None
                and worker.unit_id == builder.unit_id):
            economy.move_or_repair_wall(
                turn, worker, claimed, commands,
                urgent_only=worker.unit_id in MEMORY.returning_roles,
                memory=MEMORY,
            )
        if (worker.unit_id in MEMORY.returning_roles and worker.unit_id not in commands
                and not _inside_defense(turn, worker)):
            _return_home(turn, worker, claimed, commands)

    if day_round >= 60:
        # 夜幕将至：建造工回双炮位，矿工继续外部现金流，开拓者回单炮位。
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
        )
        _pre_night_fortify(turn, defenders, plan, claimed, claimed_sites, commands)
    else:
        # 首日把全部 75 金币用于三炮，正常成长采购从第二天开始；此后升级优先于
        # 扩展墙。危墙和濒危基地已经由上面的 emergency_want 抢占处理。
        if (P.day_index(turn.round_no) >= 2 and len(turn.weapons()) >= 3
                and not has_carried_upgrade):
            _dispatch_shopping(turn, workers, claimed, commands)
        for worker in workers:
            if worker.unit_id not in commands and worker.unit_id not in MEMORY.returning_roles:
                economy.worker_day(
                    turn, worker, plan, claimed, claimed_sites, MEMORY, commands,
                    income_only=(night_miner is not None
                                 and worker.unit_id == night_miner.unit_id),
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
            if len(turn.weapons()) >= 3 and MEMORY.gunner_pos is not None:
                # 直接以固定炮位为返程终点，避免先回基地再反向走到炮位。
                _stage_gunners(turn, (pioneer,), claimed, commands)
            else:
                _return_home(turn, pioneer, claimed, commands)

    if day_round >= 60:
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
    """编号较小的工人固定负责昼夜采矿；另一名工人负责建造、购物和双炮位。"""
    workers = turn.workers()
    if len(workers) < 2:
        return None
    miner = workers[0]
    # 只有矿工自己误拿了升级券才暂停采矿；建造工持券不能拖停矿工。
    if any(miner.has(name) for name in UPGRADE_ITEMS):
        return None
    return miner


def _builder_operator(turn: P.Turn) -> Unit | None:
    """两工人局由编号较大的工人承担建造、采购、维修和双炮轮射。"""
    workers = turn.workers()
    if not workers:
        return None
    return workers[-1]


def _home_distance(turn: P.Turn, role: Unit) -> int:
    station = turn.station()
    if station is None:
        return 0
    if (role.kind == P.PIONEER and len(turn.weapons()) >= 3
            and MEMORY.gunner_pos is not None):
        if role.pos == MEMORY.gunner_pos:
            return 0
        path = path_to_any(
            turn, role, {MEMORY.gunner_pos}, reserved=MEMORY.failed_move_cells(role.unit_id),
        )
    else:
        if _inside_defense(turn, role):
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


def _gate_position(turn: P.Turn) -> Pos | None:
    layout = _oriented_defense_layout(turn)
    if layout is None:
        return None
    MEMORY.gate_pos = layout[5]
    MEMORY.service_gate_pos = layout[6]
    return MEMORY.gate_pos


def _service_gate_position(turn: P.Turn) -> Pos | None:
    _gate_position(turn)
    return MEMORY.service_gate_pos


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
    layout = _oriented_defense_layout(turn)
    if layout is None:
        return ()
    # 初始 C 墙的前 6 格是一整面迎敌墙，任何侧翼扩展都不能抢在它们之前。
    return tuple(pos for pos in layout[3][:6]
                 if turn.land(pos) and MEMORY.build_allowed(pos, WALL))


def _move_reserved(role: Unit, claimed: set[Pos]) -> set[Pos]:
    reserved = set(claimed) | MEMORY.failed_move_cells(role.unit_id)
    if (role.kind == P.WORKER and MEMORY.gunner_pos is not None
            and role.pos != MEMORY.gunner_pos):
        reserved.add(MEMORY.gunner_pos)
    return reserved


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
        # 最后 5 回合完全留给固定炮位归位，不能再因补一面远墙错过夜战。
        if day_round >= 66:
            continue
        if not _inside_defense(turn, worker):
            step = next_step_adjacent_to_any(
                turn, worker, list(footprint), reserved=_move_reserved(worker, claimed),
            )
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
    """开拓者守单炮位，建造工守可覆盖两炮的位置并兼顾维修。"""
    available_roles = [role for role in roles if role.unit_id not in commands]
    pioneer = next((role for role in available_roles if role.kind == P.PIONEER), None)
    assignments: list[tuple[Unit, Pos]] = []
    if pioneer is not None and MEMORY.gunner_pos is not None:
        assignments.append((pioneer, MEMORY.gunner_pos))
    builder = _builder_operator(turn)
    repairer = next(
        (role for role in available_roles
         if builder is not None and role.unit_id == builder.unit_id),
        None,
    )
    if repairer is not None and MEMORY.repair_gunner_pos is not None:
        assignments.append((repairer, MEMORY.repair_gunner_pos))
    for role, target in assignments:
        if role.pos != target:
            step = next_step_to_any(
                turn, role, {target}, reserved=_move_reserved(role, claimed),
            )
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[role.unit_id] = move_command(step)
    if assignments:
        return

    # 角色或缓存不完整时仍按炮台就近值守，不能因为布局记忆丢失而整夜不开火。
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
        # 升级券已有持有者时直接交付；修复包等可叠加消耗品的 num 表示“还需购买
        # 的数量”，不能因为背包已有 1 个就错误跳过补库存。
        holder = next(
            (w for w in workers if name in carried_upgrades and w.count(name) >= num),
            None,
        )
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
                  else list(reversed(workers)) if name in maintenance
                  else list(reversed(workers)))
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
        front = set(_front_wall_sites(turn))
        central = {
            wall.pos: index
            for index, wall in enumerate(economy.central_front_walls(turn))
        }
        wall = min(
            (w for w in turn.walls() if w.level == wanted_level),
            key=lambda item: (
                0 if item.pos in central else 1 if item.pos in front else 2,
                central.get(item.pos, 99),
                item.health / economy._wall_max_health(item.level),
                -_enemy_projection(turn, item.pos),
                item.unit_id,
            ),
            default=None,
        )
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
    layout = _oriented_defense_layout(turn)
    if layout is None:
        return economy.Plan()
    target_towers, primary, repair, initial_walls, expanded_walls, gate, service_gate = layout
    MEMORY.gate_pos = gate
    MEMORY.service_gate_pos = service_gate
    MEMORY.gunner_pos = primary
    MEMORY.repair_gunner_pos = repair
    MEMORY.tower_layout = target_towers

    # 三座火箭全部位于背敌侧。它们离迎敌墙最远也只有 4~5 格，一级射程即可
    # 覆盖整条正面；两处固定炮位分别覆盖 1 炮和 2 炮。
    occupied = turn.occupied_cells() - {role.pos for role in turn.controllable()}
    needed = max(0, 3 - len(turn.weapons()))
    existing = {tower.pos for tower in turn.weapons()}
    candidates = [
        pos for pos in target_towers
        if pos not in existing and pos not in occupied
        and turn.land(pos) and MEMORY.build_allowed(pos, P.ROCKET)
    ][:needed]
    tower_sites = tuple(candidates)
    tower_kinds = tuple(economy.TOWER_LOADOUT[:len(tower_sites)])

    # 先完成 10 面连续 C 墙。外圈扩到 18 面是后期富余项：不能在第二天
    # 继续抢占建造工，拖慢三炮升级、基地回血保险和正面墙升级。两个背面
    # 通道始终排除在计划之外，任何回合都不会生成 remove 指令。
    planned_walls = expanded_walls if _expansion_ready(
        turn, initial_walls[:6],
    ) else initial_walls
    wall_sites = tuple(
        pos for pos in planned_walls
        if turn.land(pos) and MEMORY.build_allowed(pos, WALL)
    )
    return economy.Plan(
        tower_sites=tower_sites, tower_kinds=tower_kinds, wall_sites=wall_sites,
    )


def _expansion_ready(turn: P.Turn, front_sites: tuple[Pos, ...]) -> bool:
    """只在核心防线已经稳定时扩建外圈。

    门槛来自实战日志中的闭环：第四天、三门三级炮、正面六墙二级且健康，
    并保留 250 金币应急。不满足时建造工转去赚钱、修墙和送升级券。
    """
    if P.day_index(turn.round_no) < 4 or turn.gold < 250:
        return False
    weapons = sorted(turn.weapons(), key=lambda weapon: weapon.level, reverse=True)
    if len(weapons) < 3 or any(weapon.level < 3 for weapon in weapons[:3]):
        return False
    if any(
        weapon.health < economy._wall_max_health(weapon.level) * economy.HEALTHY_WALL_RATIO
        for weapon in weapons[:3]
    ):
        return False
    walls_by_pos = {wall.pos: wall for wall in turn.walls()}
    return all(
        (wall := walls_by_pos.get(pos)) is not None
        and wall.level >= 2
        and wall.health >= economy._wall_max_health(wall.level) * economy.HEALTHY_WALL_RATIO
        for pos in front_sites
    )


def _oriented_defense_layout(
    turn: P.Turn,
) -> tuple[
    tuple[Pos, ...], Pos, Pos, tuple[Pos, ...], tuple[Pos, ...], Pos, Pos
] | None:
    """生成可镜像到四角的背面三炮 + 连续 C 墙布局。"""
    station = turn.station()
    if station is None:
        return None
    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    base_x, base_y = _base_center(turn)
    enemy_x, enemy_y = _enemy_anchor(turn)
    delta_x, delta_y = enemy_x - base_x, enemy_y - base_y

    outer = set(_ring(footprint, 2))
    if abs(delta_x) >= abs(delta_y):
        front_sign = 1 if delta_x >= 0 else -1
        secondary_sign = 1 if delta_y > 0 else -1
        front_x = xmax + 2 if front_sign > 0 else xmin - 2
        rear_x = xmin - 1 if front_sign > 0 else xmax + 1
        operator_x = xmin if front_sign > 0 else xmax
        outer_rear_x = xmin - 2 if front_sign > 0 else xmax + 2

        if secondary_sign < 0:
            tower_axis = (ymax + 1, ymin, ymin - 1)
            primary_axis, repair_axis, omitted_axis = ymax + 1, ymin - 1, ymax
        else:
            tower_axis = (ymin - 1, ymax, ymax + 1)
            primary_axis, repair_axis, omitted_axis = ymin - 1, ymax + 1, ymin
        towers = tuple(Pos(rear_x, value) for value in tower_axis)
        primary = Pos(operator_x, primary_axis)
        repair = Pos(operator_x, repair_axis)
        gate = Pos(outer_rear_x, omitted_axis)
        service_gate = Pos(outer_rear_x, repair_axis)

        center = (ymin + ymax) / 2
        front_wall = sorted(
            (Pos(front_x, value) for value in range(ymin - 2, ymax + 3)),
            key=lambda pos: (abs(pos.y - center), -secondary_sign * pos.y),
        )
        inward = (front_x - front_sign, front_x - 2 * front_sign)
        flank_wall = tuple(
            Pos(value, edge)
            for value in inward
            for edge in (ymax + 2, ymin - 2)
        )
    else:
        front_sign = 1 if delta_y >= 0 else -1
        secondary_sign = 1 if delta_x > 0 else -1
        front_y = ymax + 2 if front_sign > 0 else ymin - 2
        rear_y = ymin - 1 if front_sign > 0 else ymax + 1
        operator_y = ymin if front_sign > 0 else ymax
        outer_rear_y = ymin - 2 if front_sign > 0 else ymax + 2

        if secondary_sign < 0:
            tower_axis = (xmax + 1, xmin, xmin - 1)
            primary_axis, repair_axis, omitted_axis = xmax + 1, xmin - 1, xmax
        else:
            tower_axis = (xmin - 1, xmax, xmax + 1)
            primary_axis, repair_axis, omitted_axis = xmin - 1, xmax + 1, xmin
        towers = tuple(Pos(value, rear_y) for value in tower_axis)
        primary = Pos(primary_axis, operator_y)
        repair = Pos(repair_axis, operator_y)
        gate = Pos(omitted_axis, outer_rear_y)
        service_gate = Pos(repair_axis, outer_rear_y)

        center = (xmin + xmax) / 2
        front_wall = sorted(
            (Pos(value, front_y) for value in range(xmin - 2, xmax + 3)),
            key=lambda pos: (abs(pos.x - center), -secondary_sign * pos.x),
        )
        inward = (front_y - front_sign, front_y - 2 * front_sign)
        flank_wall = tuple(
            Pos(edge, value)
            for value in inward
            for edge in (xmax + 2, xmin - 2)
        )

    initial = tuple(front_wall) + flank_wall
    gaps = {gate, service_gate}
    remainder = sorted(
        outer - set(initial) - gaps,
        key=lambda pos: (-_enemy_projection(turn, pos), pos.x, pos.y),
    )
    expanded = initial + tuple(remainder)
    return towers, primary, repair, initial, expanded, gate, service_gate


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
    central_walls = economy.central_front_walls(turn)
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    if day_round in {1, 50, 60} or 70 <= day_round <= 80 or turn.round_no % 10 == 0:
        LOGGER.info(
            "defense round=%d station_hp=%s walls=%d towers=%d workers=%d "
            "gold=%d stone=%d robots=%d ready_towers=%d attacks=%d station_level=%d "
            "central_walls=%s score=%d",
            turn.round_no, station.health if station else 0, len(turn.walls()),
            len(turn.weapons()), len(turn.workers()), turn.gold,
            sum(worker.count(P.STONE) for worker in turn.workers()), len(turn.robots),
            sum(any(distance(role.pos, tower.pos) <= 1
                    for role in turn.controllable()) for tower in turn.weapons()),
            sum(command.get("action") == "attack" for command in commands.values()),
            station.level if station else 0,
            [(wall.pos.x, wall.pos.y, wall.level, wall.health) for wall in central_walls],
            turn.total_score,
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
    for key, command in list(commands.items()):
        if command.get("action") == "attack":
            role_id = int(command["controllerId"])
            if role_id in commands or role_id in controllers:
                commands.pop(key)
            else:
                controllers.add(role_id)
    # 先去掉振荡和多人争抢同一终点，再允许“前车已经驶离”的跟随移动。
    # 旧逻辑把所有当前角色格都视为永久占用，狭窄后门中会把合法队列截断，继而
    # 触发下一回合反向寻路，形成 A-B-A-B 往返。
    moves: dict[int, Pos] = {}
    destinations: dict[Pos, int] = {}
    for key, command in list(commands.items()):
        if command.get("action") != "move":
            continue
        role_id = int(key)
        pos = Pos.load(command["targetPos"][0])
        if MEMORY.would_oscillate(role_id, pos):
            LOGGER.warning("oscillation stopped round=%d role=%d target=%s",
                           turn.round_no, role_id, pos)
            MEMORY.note_move_result(role_id, pos, False)
            commands.pop(key)
            continue
        if pos in destinations:
            commands.pop(key)
            continue
        destinations[pos] = role_id
        moves[role_id] = pos

    occupants = {role.pos: role.unit_id for role in turn.controllable()}
    changed = True
    while changed:
        changed = False
        for role_id, destination in list(moves.items()):
            occupant = occupants.get(destination)
            if occupant is None or occupant == role_id:
                continue
            occupant_destination = moves.get(occupant)
            # 静止占位、直接换位和已被删除的移动都不安全；向正在离开的角色后方
            # 跟随则保留，使两名角色可以在窄通道中同向前进。
            blocked = occupant_destination is None
            if occupant_destination == next(
                (role.pos for role in turn.controllable() if role.unit_id == role_id),
                None,
            ):
                blocked = True
            if blocked:
                commands.pop(role_id, None)
                moves.pop(role_id, None)
                changed = True
    active_destinations = set(moves.values())
    for key, command in list(commands.items()):
        if (command.get("action") == "build"
                and Pos.load(command["targetPos"][0]) in active_destinations):
            commands.pop(key)

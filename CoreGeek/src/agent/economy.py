"""白天经济: 选矿采集、批量卖矿、建造围墙、商店购物(升级/修复)。

设计要点:
- 工人以"收益 = 价格 / 往返步数"选矿, 并结合新闻推断的停产预期(推理类任务联动)。
- 达到批量阈值或背包将满时去小贩处贩卖(每回合仅一个动作)。
- 金币支出优先级: 基地升级券 > 武器升级券 > 围墙升级/修复; 常备少量石头。
"""
from dataclasses import dataclass
from typing import Any

from . import protocol as P
from .grid import neighbours, next_step_adjacent, next_step_adjacent_to_any
from .memory import Memory
from .protocol import (
    COPPER,
    GATLING,
    IRON,
    Pos,
    RAILGUN,
    ROCKET,
    STATION_UP_V1,
    STATION_UP_V2,
    STONE,
    Unit,
    WALL,
    WALL_FIXER,
    WALL_UP_V1,
    WALL_UP_V2,
    WEAPON_UP_V1,
    WEAPON_UP_V2,
    build_command,
    buy_command,
    collect_command,
    distance,
    move_command,
    sell_command,
    station_footprint,
    use_command,
)

TOWER_LOADOUT = (ROCKET, ROCKET, ROCKET)
WALL_STONE_KEEP = 2  # 工人背包中常备的墙材料石头数
STONE_BUILD_BATCH = 5  # 避免每采一块石头就长途往返基地
HEALTHY_WALL_RATIO = 0.8  # 白天尽早修墙，避免残血墙进入夜晚
SELL_BATCH = 8
NIGHT_ROBOT_CLEARANCE = 6


@dataclass
class Plan:
    """白天建造计划(由主决策模块生成, 经济模块消费)。"""

    tower_sites: tuple[Pos, ...] = ()
    tower_kinds: tuple[str, ...] = ()
    wall_sites: tuple[Pos, ...] = ()


def worker_day(
    turn: P.Turn,
    worker: Unit,
    plan: Plan,
    claimed: set[Pos],
    claimed_sites: set[Pos],
    memory: Memory,
    commands: dict[int, dict[str, Any]],
    *,
    income_only: bool = False,
    force_stone: bool = False,
) -> None:
    """建造工补防线，矿工专注高价值矿和现金流。"""
    day = P.day_index(turn.round_no)
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    if income_only:
        if _try_sell(turn, worker, day, memory, claimed, commands, cash_first=True, stone_keep=0):
            return
        if not worker.backpack_full and _try_collect_income(
            turn, worker, day, memory, claimed, commands,
        ):
            return
        if day_round < 55:
            _hold_near_station(turn, worker, memory, claimed, commands)
        return

    walls_needed = any(
        site not in turn.occupied_cells() and memory.build_allowed(site, WALL)
        for site in plan.wall_sites
    )
    towers_needed = bool(plan.tower_sites and turn.gold >= P.WEAPON_BUILD_COST)
    stone_count = worker.count(STONE)

    if force_stone and stone_count == 0:
        if worker.backpack_full:
            if _try_sell(turn, worker, day, memory, claimed, commands, stone_keep=0):
                return
        elif _try_collect_stone(turn, worker, memory, claimed, commands):
            return

    # 持续消耗整批石头，不能建一块后又返回矿点补到阈值。
    mode = memory.worker_modes.setdefault(worker.unit_id, "build" if stone_count else "collect")
    if stone_count == 0:
        mode = "collect"
    if stone_count >= STONE_BUILD_BATCH or day_round >= 48:
        mode = "build"
    memory.worker_modes[worker.unit_id] = mode
    if (walls_needed and not towers_needed and mode == "collect"
            and not worker.backpack_full
            and _try_collect_stone(turn, worker, memory, claimed, commands)):
        return

    # 1) 建造优先(武器 > 围墙), 建造是白天限时机会
    if _try_build(turn, worker, plan, claimed, claimed_sites, memory, commands):
        return

    # 未完成防线时，石头是必需建材；不按市场价格把它排在铁/铜之后。
    if walls_needed and stone_count == 0:
        if worker.backpack_full:
            if _try_sell(turn, worker, day, memory, claimed, commands):
                return
        elif _try_collect_stone(turn, worker, memory, claimed, commands):
            return

    # 2) 矿石达到批量阈值或背包将满时再卖，避免每采一个就横穿地图
    if _try_sell(turn, worker, day, memory, claimed, commands):
        return

    # 3) 背包未满则采集
    if not worker.backpack_full and _try_collect(turn, worker, day, memory, claimed, commands):
        return

    # 4) 没事做: 待在基地附近待命
    _hold_near_station(turn, worker, memory, claimed, commands)


def worker_night_safe(
    turn: P.Turn,
    worker: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """B 工夜间持续采矿；机器人接近时先撤离，绝不为一块矿冒险。"""
    def clearance(pos: Pos) -> int:
        return min((distance(pos, robot.pos) for robot in turn.robots), default=99)

    def retreat() -> bool:
        blocked = turn.blocked(worker)
        candidates = [
            pos for pos in neighbours(worker.pos)
            if turn.land(pos) and pos not in blocked
            and pos not in _move_reserved(memory, worker, claimed)
        ]
        if not candidates:
            return False
        step = max(candidates, key=lambda pos: (clearance(pos), -distance(pos, worker.pos), -pos.x, -pos.y))
        if clearance(step) <= clearance(worker.pos):
            return False
        claimed.add(step)
        commands[worker.unit_id] = move_command(step)
        memory.lock_worker_target(worker.unit_id, "retreat", step)
        return True

    if turn.robots and clearance(worker.pos) < NIGHT_ROBOT_CLEARANCE:
        return retreat()

    day = P.day_index(turn.round_no)
    trial_claimed = set(claimed)
    trial_commands: dict[int, dict[str, Any]] = {}
    acted = _try_sell(turn, worker, day, memory, trial_claimed, trial_commands)
    if not acted and not worker.backpack_full:
        acted = _try_collect_income(
            turn, worker, day, memory, trial_claimed, trial_commands,
        )
    command = trial_commands.get(worker.unit_id)
    if not acted or command is None:
        return False
    # sell/buy 指令没有 targetPos；只有移动指令才会改变暴露位置。
    target_raw = command.get("targetPos")
    exposed = (Pos.load(target_raw[0])
               if command["action"] == "move" and target_raw else worker.pos)
    if turn.robots and clearance(exposed) < NIGHT_ROBOT_CLEARANCE:
        return retreat()
    claimed.update(trial_claimed)
    commands[worker.unit_id] = command
    return True


def _try_build(
    turn: P.Turn,
    worker: Unit,
    plan: Plan,
    claimed: set[Pos],
    claimed_sites: set[Pos],
    memory: Memory,
    commands: dict[int, dict[str, Any]],
) -> bool:
    standing = turn.occupied_cells()
    jobs: list[tuple[Pos, str]] = []
    for i, site in enumerate(plan.tower_sites):
        kind = plan.tower_kinds[i] if i < len(plan.tower_kinds) else TOWER_LOADOUT[i % 3]
        reserved_gold = sum(
            P.WEAPON_BUILD_COST
            for cmd in commands.values()
            if cmd.get("action") == "build" and cmd.get("name") in P.TOWER_TYPES
        )
        if (
            site not in standing
            and turn.gold - reserved_gold >= P.WEAPON_BUILD_COST
            and memory.build_allowed(site, kind)
        ):
            jobs.append((site, kind))
    for site in plan.wall_sites:
        if site not in standing and worker.count(P.WALL_MATERIAL) > 0 and memory.build_allowed(site, WALL):
            jobs.append((site, WALL))

    locked = memory.worker_targets.get(worker.unit_id)
    if memory.worker_target_kinds.get(worker.unit_id) == "build" and locked is not None:
        jobs.sort(key=lambda job: (job[0] != locked, distance(worker.pos, job[0])))

    for site, kind in jobs:
        if site in claimed_sites:
            continue
        if distance(worker.pos, site) <= 1:
            commands[worker.unit_id] = build_command(site, kind)
            claimed_sites.add(site)
            claimed.add(site)
            memory.clear_worker_target(worker.unit_id)
            return True
        step = next_step_adjacent(
            turn, worker, site, reserved=_move_reserved(memory, worker, claimed),
        )
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[worker.unit_id] = move_command(step)
            claimed_sites.add(site)
            memory.lock_worker_target(worker.unit_id, "build", site)
            return True
    if memory.worker_target_kinds.get(worker.unit_id) == "build":
        memory.clear_worker_target(worker.unit_id)
    return False


def _try_sell(
    turn: P.Turn,
    worker: Unit,
    day: int,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    cash_first: bool = False,
    stone_keep: int = WALL_STONE_KEEP,
) -> bool:
    vendor = turn.vendor_pos()
    if vendor is None:
        return False
    # 选最值钱的非石头矿石; 石头仅超出常备量才卖
    best_ore, best_num, best_value = None, 0, 0
    for ore in (COPPER, IRON, STONE):
        # 明日停产而今日仍可采时先囤货，等价格上涨后再卖。
        if (not cash_first and turn.gold >= 150
                and not memory.ore_blocked(ore, day) and memory.ore_blocked(ore, day + 1)):
            continue
        num = worker.count(ore)
        if ore == STONE:
            num = max(0, num - stone_keep)
        price = turn.vendor_prices.get(ore, 0)
        if num > 0 and num * price > best_value:
            best_ore, best_num, best_value = ore, num, num * price
    if best_ore is None:
        return False
    total_sellable = sum(
        max(0, worker.count(ore) - (stone_keep if ore == STONE else 0))
        for ore in (COPPER, IRON, STONE)
    )
    nearly_full = worker.capacity is not None and len(worker.backpack) >= int(worker.capacity * 0.8)
    needs_cash = cash_first and (
        turn.gold < 100 <= turn.gold + best_value
        or (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1 >= 45
    )
    if total_sellable < SELL_BATCH and not nearly_full and not needs_cash:
        return False
    if distance(worker.pos, vendor) <= 1:
        commands[worker.unit_id] = sell_command(best_ore, best_num)
        memory.clear_worker_target(worker.unit_id)
        return True
    step = next_step_adjacent(
        turn, worker, vendor, reserved=_move_reserved(memory, worker, claimed),
    )
    if step is not None and step not in claimed:
        claimed.add(step)
        commands[worker.unit_id] = move_command(step)
        memory.lock_worker_target(worker.unit_id, "sell", vendor)
        return True
    return False


def _try_collect(
    turn: P.Turn,
    worker: Unit,
    day: int,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    def value(pos: Pos) -> tuple[float, int]:
        ore = turn.zones.get(pos, "")
        price = turn.vendor_prices.get(ore, 0)
        if memory.ore_blocked(ore, day):
            price = 0
        elif memory.ore_blocked(ore, day + 1):
            price *= 3  # 即将停产，优先抢采
        if ore == STONE:
            price = max(price, 2)  # 石头有建造价值, 给保底价
        travel = max(1, distance(worker.pos, pos))
        return (-(price / travel), travel)

    candidates = sorted(
        (p for p in _all_mine_positions(turn) if p not in claimed),
        key=value,
    )
    locked = memory.worker_targets.get(worker.unit_id)
    if memory.worker_target_kinds.get(worker.unit_id) == "collect" and locked in candidates:
        candidates.remove(locked)
        candidates.insert(0, locked)
    for mine in candidates:
        if distance(worker.pos, mine) <= 1:
            claimed.add(mine)
            commands[worker.unit_id] = collect_command(mine)
            memory.lock_worker_target(worker.unit_id, "collect", mine)
            return True
        step = next_step_adjacent(
            turn, worker, mine, reserved=_move_reserved(memory, worker, claimed),
        )
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[worker.unit_id] = move_command(step)
            memory.lock_worker_target(worker.unit_id, "collect", mine)
            return True
    return False


def _all_mine_positions(turn: P.Turn) -> list[Pos]:
    return [pos for pos, kind in turn.zones.items() if kind in (STONE, IRON, COPPER)]


def _try_collect_income(
    turn: P.Turn,
    worker: Unit,
    day: int,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """矿工优先铜/铁，避免因为近处石矿而整天没有金币收入。"""
    mines = [
        pos for pos, ore in turn.zones.items()
        if ore in (COPPER, IRON) and not memory.ore_blocked(ore, day)
    ]
    vendor = turn.vendor_pos() or worker.pos
    mines.sort(key=lambda pos: -turn.vendor_prices.get(turn.zones[pos], 0) /
               (distance(worker.pos, pos) + distance(pos, vendor) + SELL_BATCH))
    locked = memory.worker_targets.get(worker.unit_id)
    if memory.worker_target_kinds.get(worker.unit_id) == "income" and locked in mines:
        mines.remove(locked)
        mines.insert(0, locked)
    for mine in mines:
        if distance(worker.pos, mine) <= 1:
            commands[worker.unit_id] = collect_command(mine)
            memory.lock_worker_target(worker.unit_id, "income", mine)
            return True
        step = next_step_adjacent(
            turn, worker, mine, reserved=_move_reserved(memory, worker, claimed),
        )
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[worker.unit_id] = move_command(step)
            memory.lock_worker_target(worker.unit_id, "income", mine)
            return True
    return _try_collect(turn, worker, day, memory, claimed, commands)


def _try_collect_stone(
    turn: P.Turn,
    worker: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """防线缺口优先补石头；同一矿点允许多工人同时采集。"""
    mines = sorted(turn.mines_of(STONE), key=lambda pos: distance(worker.pos, pos))
    locked = memory.worker_targets.get(worker.unit_id)
    if memory.worker_target_kinds.get(worker.unit_id) == "stone" and locked in mines:
        mines.remove(locked)
        mines.insert(0, locked)
    for mine in mines:
        if distance(worker.pos, mine) <= 1:
            commands[worker.unit_id] = collect_command(mine)
            memory.lock_worker_target(worker.unit_id, "stone", mine)
            return True
        step = next_step_adjacent(
            turn, worker, mine, reserved=_move_reserved(memory, worker, claimed),
        )
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[worker.unit_id] = move_command(step)
            memory.lock_worker_target(worker.unit_id, "stone", mine)
            return True
    return False


def _hold_near_station(
    turn: P.Turn,
    worker: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    station = turn.station()
    if station is None:
        return
    footprint = station_footprint(station.pos)
    target = footprint[2]  # 基地左下格附近
    if distance(worker.pos, target) <= 1:
        return
    step = next_step_adjacent_to_any(
        turn, worker, list(footprint), reserved=_move_reserved(memory, worker, claimed),
    )
    if step is not None and step not in claimed:
        claimed.add(step)
        commands[worker.unit_id] = move_command(step)


def _move_reserved(memory: Memory, worker: Unit, claimed: set[Pos]) -> set[Pos]:
    return set(claimed) | memory.failed_move_cells(worker.unit_id)


def shopping_list(turn: P.Turn) -> list[tuple[str, int]]:
    """先形成火力/基地成长闭环；消耗品只处理真正紧急的缺口。"""
    items: list[tuple[str, int]] = []
    gold = turn.gold
    station = turn.station()
    def add(name: str, price: int) -> bool:
        nonlocal gold
        price = turn.shop_prices.get(name, price)
        if gold < price or any(role.has(name) for role in turn.controllable()):
            return False
        items.append((name, 1))
        gold -= price
        return True

    urgent_station = station is not None and station.health < 1500 * max(1, station.level) * 0.65
    station_item = (STATION_UP_V1, 100) if station and station.level == 1 else (STATION_UP_V2, 150)
    if urgent_station and station.level < 3:
        add(*station_item)
    walls = turn.walls()
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    has_repair_kit = any(worker.has(WALL_FIXER) for worker in turn.workers())
    critical_wall = any(
        wall.health < _wall_max_health(wall.level) * 0.35
        for wall in walls
    )
    if walls and not has_repair_kit and critical_wall:
        add(WALL_FIXER, 10)

    # 三炮成型后先把基地升二级，再扩第一门火箭；先抬高生存线，避免后期被秒基地。
    weapons = turn.weapons()
    level1_weapon = next((weapon for weapon in weapons if weapon.level == 1), None)
    has_level2_weapon = any(weapon.level >= 2 for weapon in weapons)
    if station is not None and station.level == 1 and len(weapons) >= 3 and not urgent_station:
        add(STATION_UP_V1, 100)
    if level1_weapon is not None and not has_level2_weapon:
        add(WEAPON_UP_V1, 100)
    if level1_weapon is not None and has_level2_weapon:
        add(WEAPON_UP_V1, 100)

    if any(worker.health < 110 for worker in turn.workers()):
        add(P.MEDICINE, 10)
    # 非紧急修复包不能反复吞掉升级储蓄；升级预算之外有余钱时再购买。
    growth_pending = (level1_weapon is not None
                      or station is not None and station.level == 1)
    if (walls and not has_repair_kit and not critical_wall
            and (not growth_pending or gold >= 110)
            and (day_round >= 55 or any(
                wall.health < _wall_max_health(wall.level) * HEALTHY_WALL_RATIO
                for wall in walls
            ))):
        add(WALL_FIXER, 10)
    level2_weapon = next((weapon for weapon in turn.weapons() if weapon.level == 2), None)
    if level2_weapon is not None:
        add(WEAPON_UP_V2, 150)
    if station is not None and station.level == 2 and not urgent_station:
        add(STATION_UP_V2, 150)
    level1_wall = next((wall for wall in turn.walls() if wall.level == 1), None)
    if station is not None and station.level >= 2 and level1_wall is not None and gold >= 120:
        items.append((WALL_UP_V1, 1))
        gold -= 20
    level2_wall = next((wall for wall in turn.walls() if wall.level == 2), None)
    if station is not None and station.level >= 2 and level2_wall is not None and gold >= 130:
        items.append((WALL_UP_V2, 1))
        gold -= 30
    return items


def use_repair_if_needed(turn: P.Turn, role: Unit, commands: dict[int, dict[str, Any]]) -> bool:
    """角色身边有残血墙且背包有修复包 -> 使用。"""
    if not role.has(WALL_FIXER):
        return False
    for wall in turn.walls():
        if wall.health < _wall_max_health(wall.level) * HEALTHY_WALL_RATIO and distance(role.pos, wall.pos) <= 1:
            commands[role.unit_id] = use_command(WALL_FIXER, wall.pos)
            return True
    return False


def move_or_repair_wall(
    turn: P.Turn,
    role: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    urgent_only: bool = False,
    allow_move: bool = True,
    memory: Memory | None = None,
) -> bool:
    """修复包已在背包时仍持续安排使用，不依赖本回合能否继续购买。"""
    if not role.has(WALL_FIXER):
        return False
    ratio = 0.35 if urgent_only and not turn.is_day else HEALTHY_WALL_RATIO
    damaged = [
        wall for wall in turn.walls()
        if (wall.health < _wall_max_health(wall.level) * ratio
            or wall.health <= 2 * incoming_wall_damage(turn, wall))
        and (not urgent_only and allow_move or distance(role.pos, wall.pos) <= 1)
        and wall.pos not in claimed
    ]
    if not damaged:
        return False
    wall = min(damaged, key=lambda item: (
        distance(role.pos, item.pos) > 1,
        item.health / max(1, incoming_wall_damage(turn, item)),
        item.health / _wall_max_health(item.level), distance(role.pos, item.pos),
    ))
    if distance(role.pos, wall.pos) <= 1:
        commands[role.unit_id] = use_command(WALL_FIXER, wall.pos)
        claimed.add(wall.pos)
        if memory is not None:
            memory.clear_worker_target(role.unit_id)
        return True
    if urgent_only or not allow_move:
        return False  # 夜间不要为了远处的墙放弃炮位
    reserved = claimed if memory is None else _move_reserved(memory, role, claimed)
    step = next_step_adjacent(turn, role, wall.pos, reserved=reserved)
    if step is not None and step not in claimed:
        claimed.add(step)
        commands[role.unit_id] = move_command(step)
        claimed.add(wall.pos)
        if memory is not None:
            memory.lock_worker_target(role.unit_id, "repair", wall.pos)
        return True
    return False


def _wall_max_health(level: int) -> int:
    return (1000, 1500, 2000)[min(max(level, 1), 3) - 1]


def incoming_wall_damage(turn: P.Turn, wall: Unit) -> int:
    powers = {P.SMALL_ROBOT: 5, P.MIDDLE_ROBOT: 10, P.LARGE_ROBOT: 20, P.BOSS_ROBOT: 40}
    return sum(powers.get(robot.kind, 5) for robot in turn.robots
               if distance(robot.pos, wall.pos) <= 3
               and (not robot.target_team or robot.target_team == turn.team_type))

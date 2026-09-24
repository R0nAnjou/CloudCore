"""白天经济: 选矿采集、批量卖矿、建造围墙、商店购物(升级/修复)。

设计要点:
- 工人以"收益 = 价格 / 往返步数"选矿, 并结合新闻推断的停产预期(推理类任务联动)。
- 达到批量阈值或背包将满时去小贩处贩卖(每回合仅一个动作)。
- 金币支出优先级: 三炮最低核心 > 迎敌中央双墙三级 > 基地保险 > 其余成长。
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
NIGHT_ROBOT_RELEASE_CLEARANCE = 8
NIGHT_RETREAT_HOLD_ROUNDS = 4


@dataclass
class Plan:
    """白天建造计划(由主决策模块生成, 经济模块消费)。"""

    tower_sites: tuple[Pos, ...] = ()
    tower_kinds: tuple[str, ...] = ()
    wall_sites: tuple[Pos, ...] = ()


def central_front_walls(turn: P.Turn) -> tuple[Unit, ...]:
    """返回迎敌正面最中间的两面已建围墙。

    先按敌方所在的主方向取最外侧墙线，再按与基地中心的横向偏移取中间两格。
    这与四个出生角的镜像 C 墙布局兼容。
    """
    station = turn.station()
    walls = list(turn.walls())
    if station is None or len(walls) < 2:
        return ()
    footprint = station_footprint(station.pos)
    base_x = sum(pos.x for pos in footprint) / len(footprint)
    base_y = sum(pos.y for pos in footprint) / len(footprint)
    enemy_station = next(
        (role for role in turn.enemy_roles if role.kind == P.STATION), None,
    )
    if enemy_station is not None:
        enemy_cells = station_footprint(enemy_station.pos)
        enemy_x = sum(pos.x for pos in enemy_cells) / len(enemy_cells)
        enemy_y = sum(pos.y for pos in enemy_cells) / len(enemy_cells)
    elif turn.enemy_roles:
        enemy_x = sum(role.pos.x for role in turn.enemy_roles) / len(turn.enemy_roles)
        enemy_y = sum(role.pos.y for role in turn.enemy_roles) / len(turn.enemy_roles)
    else:
        enemy_x, enemy_y = turn.width / 2, turn.height / 2
    delta_x, delta_y = enemy_x - base_x, enemy_y - base_y
    if abs(delta_x) >= abs(delta_y):
        edge = (max(wall.pos.x for wall in walls) if delta_x >= 0
                else min(wall.pos.x for wall in walls))
        front = [wall for wall in walls if wall.pos.x == edge]
        front.sort(key=lambda wall: (abs(wall.pos.y - base_y), wall.pos.y, wall.unit_id))
    else:
        edge = (max(wall.pos.y for wall in walls) if delta_y >= 0
                else min(wall.pos.y for wall in walls))
        front = [wall for wall in walls if wall.pos.y == edge]
        front.sort(key=lambda wall: (abs(wall.pos.x - base_x), wall.pos.x, wall.unit_id))
    return tuple(front[:2]) if len(front) >= 2 else ()


def central_wall_upgrade_need(turn: P.Turn) -> tuple[str, int] | None:
    """中央双墙先同步升二级，再同步升三级。"""
    central = central_front_walls(turn)
    if len(central) < 2:
        return None
    level1 = sum(wall.level == 1 for wall in central)
    if level1:
        return WALL_UP_V1, level1
    level2 = sum(wall.level == 2 for wall in central)
    if level2:
        return WALL_UP_V2, level2
    return None


def central_wall_upgrade_reserve(turn: P.Turn) -> int:
    """第三夜前中央双墙到三级还需预留的金币。"""
    central = central_front_walls(turn)
    if len(central) < 2:
        return 0
    need_v1 = sum(wall.level == 1 for wall in central)
    need_v2 = sum(wall.level <= 2 for wall in central)
    held_v1 = sum(role.count(WALL_UP_V1) for role in turn.controllable())
    held_v2 = sum(role.count(WALL_UP_V2) for role in turn.controllable())
    price_v1 = turn.shop_prices.get(WALL_UP_V1, 20)
    price_v2 = turn.shop_prices.get(WALL_UP_V2, 30)
    return max(0, need_v1 - held_v1) * price_v1 + max(0, need_v2 - held_v2) * price_v2


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
    relevant_robots = tuple(
        robot for robot in turn.robots
        if not robot.target_team or robot.target_team == turn.team_type
        or distance(worker.pos, robot.pos) < NIGHT_ROBOT_CLEARANCE
    )

    def clearance(pos: Pos) -> int:
        return min((distance(pos, robot.pos) for robot in relevant_robots), default=99)

    def retreat() -> bool:
        blocked = turn.blocked(worker)
        candidates = [
            pos for pos in neighbours(worker.pos)
            if turn.land(pos) and pos not in blocked
            and pos not in _move_reserved(memory, worker, claimed)
        ]
        if not candidates:
            return False
        history = memory.position_history.get(worker.unit_id, [])
        previous = history[-2] if len(history) >= 2 else None
        # 被夹击时允许横向绕行；不再要求每一步都严格增加最近机器人距离。
        step = max(candidates, key=lambda pos: (
            clearance(pos),
            sum(distance(pos, robot.pos) for robot in relevant_robots),
            pos != previous,
            -pos.x,
            -pos.y,
        ))
        claimed.add(step)
        commands[worker.unit_id] = move_command(step)
        memory.lock_worker_target(worker.unit_id, "retreat", step)
        return True

    role_id = worker.unit_id
    retreating = memory.worker_modes.get(role_id) == "retreat"
    threatened = bool(relevant_robots) and clearance(worker.pos) < NIGHT_ROBOT_CLEARANCE
    if threatened:
        memory.worker_modes[role_id] = "retreat"
        memory.worker_retreat_until[role_id] = max(
            memory.worker_retreat_until.get(role_id, 0),
            turn.round_no + NIGHT_RETREAT_HOLD_ROUNDS,
        )
        memory.worker_safe_streak[role_id] = 0
        retreating = True

    if retreating and not relevant_robots:
        memory.worker_modes[role_id] = "income"
        memory.worker_retreat_until.pop(role_id, None)
        memory.worker_safe_streak.pop(role_id, None)
        memory.clear_worker_target(role_id)
        retreating = False
    if retreating:
        safely_clear = clearance(worker.pos) >= NIGHT_ROBOT_RELEASE_CLEARANCE
        if safely_clear:
            memory.worker_safe_streak[role_id] = memory.worker_safe_streak.get(role_id, 0) + 1
        else:
            memory.worker_safe_streak[role_id] = 0
        may_release = (
            turn.round_no >= memory.worker_retreat_until.get(role_id, 0)
            and memory.worker_safe_streak.get(role_id, 0) >= 2
        )
        if not may_release:
            return retreat()
        memory.worker_modes[role_id] = "income"
        memory.worker_retreat_until.pop(role_id, None)
        memory.worker_safe_streak.pop(role_id, None)
        memory.clear_worker_target(role_id)

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

    jobs.sort(key=lambda job: (job[1] == WALL, distance(worker.pos, job[0])))
    locked = memory.worker_targets.get(worker.unit_id)
    if memory.worker_target_kinds.get(worker.unit_id) == "build" and locked is not None:
        jobs.sort(key=lambda job: (job[0] != locked, job[1] == WALL,
                                   distance(worker.pos, job[0])))

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
    reserved = set(claimed) | memory.failed_move_cells(worker.unit_id)
    if worker.kind == P.WORKER and memory.gunner_pos is not None and worker.pos != memory.gunner_pos:
        reserved.add(memory.gunner_pos)
    return reserved


def shopping_list(turn: P.Turn) -> list[tuple[str, int]]:
    """主火箭三级后，第三夜前锁定中央双墙三级。"""
    items: list[tuple[str, int]] = []
    gold = turn.gold
    station = turn.station()

    def owned(name: str) -> int:
        return sum(role.count(name) for role in turn.controllable())

    def add(name: str, price: int, *, target: int = 1) -> bool:
        nonlocal gold
        price = turn.shop_prices.get(name, price)
        missing = max(0, target - owned(name)
                      - sum(num for current, num in items if current == name))
        quantity = min(missing, gold // max(1, price))
        if quantity <= 0:
            return False
        items.append((name, quantity))
        gold -= price * quantity
        return True

    urgent_station = station is not None and station.health < 1500 * max(1, station.level) * 0.50
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

    weapons = turn.weapons()
    levels = sorted((weapon.level for weapon in weapons), reverse=True)
    fire_anchor = len(levels) >= 3 and levels[0] >= 3
    core_battery = fire_anchor and levels[1] >= 2
    station_insurance = (
        station is not None and station.level < 3
        and station.health < 1500 * station.level * 0.90 and core_battery
    )
    central_need = central_wall_upgrade_need(turn)
    fortify_centre = (
        P.day_index(turn.round_no) >= 2 and fire_anchor and central_need is not None
    )

    # 真实路径：1-1-1 -> 2-1-1 -> 3-1-1，然后立即冻结其他普通成长，
    # 先把中央双墙做到 2-2，再做到 3-3；之后才继续到 3-2-1。
    if len(weapons) >= 3 and not urgent_station:
        if not fire_anchor and levels[0] < 2:
            add(WEAPON_UP_V1, 100)
        elif not fire_anchor and levels[0] < 3:
            add(WEAPON_UP_V2, 150)
        elif fortify_centre and central_need is not None:
            name, quantity = central_need
            add(name, 20 if name == WALL_UP_V1 else 30, target=quantity)
        elif not core_battery and levels[1] < 2:
            add(WEAPON_UP_V1, 100)
        elif station_insurance:
            add(
                STATION_UP_V1 if station and station.level == 1 else STATION_UP_V2,
                100 if station and station.level == 1 else 150,
            )
        elif any(weapon.level == 1 for weapon in weapons):
            add(WEAPON_UP_V1, 100)
        elif any(weapon.level == 2 for weapon in weapons):
            add(WEAPON_UP_V2, 150)

    if any(worker.health < 110 for worker in turn.workers()):
        add(P.MEDICINE, 10)
    # 非紧急修复包只使用升级后的剩余预算，逐步把全队库存补到 3 个。
    growth_pending = len(weapons) < 3 or any(weapon.level < 3 for weapon in weapons)
    if (walls and sum(worker.count(WALL_FIXER) for worker in turn.workers()) < 3
            and not critical_wall
            and (not growth_pending or gold >= 160)
            and (day_round >= 55 or any(
                wall.health < _wall_max_health(wall.level) * HEALTHY_WALL_RATIO
                for wall in walls
            ))):
        add(WALL_FIXER, 10, target=3)

    # 三炮满级后再做普通基地成长；紧急基地升级已经在函数开头处理。
    if (station is not None and station.level == 1 and not urgent_station
            and len(weapons) >= 3 and all(weapon.level >= 3 for weapon in weapons)):
        add(STATION_UP_V1, 100)
    if (station is not None and station.level == 2 and not urgent_station
            and len(weapons) >= 3 and all(weapon.level >= 3 for weapon in weapons)):
        add(STATION_UP_V2, 150)

    level1_wall = next((wall for wall in turn.walls() if wall.level == 1), None)
    if (len(levels) >= 2 and levels[0] >= 3 and levels[1] >= 3
            and level1_wall is not None and gold >= 20):
        add(WALL_UP_V1, 20)
    level2_wall = next((wall for wall in turn.walls() if wall.level == 2), None)
    if (len(levels) >= 3 and all(level >= 3 for level in levels[:3])
            and level2_wall is not None and gold >= 30):
        add(WALL_UP_V2, 30)
    return items


def emergency_shopping_list(turn: P.Turn) -> list[tuple[str, int]]:
    """不受墙数门槛限制的救命物资；不在这里安排普通成长消费。"""
    items: list[tuple[str, int]] = []
    gold = turn.gold

    def add(name: str, default_price: int) -> None:
        nonlocal gold
        price = turn.shop_prices.get(name, default_price)
        if (gold >= price
                and not any(role.has(name) for role in turn.controllable())):
            items.append((name, 1))
            gold -= price

    station = turn.station()
    if (station is not None and station.level < 3
            and station.health < 1500 * max(1, station.level) * 0.50):
        add(STATION_UP_V1 if station.level == 1 else STATION_UP_V2,
            100 if station.level == 1 else 150)

    walls = turn.walls()
    if (walls and not any(worker.has(WALL_FIXER) for worker in turn.workers())
            and any(wall.health < _wall_max_health(wall.level) * 0.35 for wall in walls)):
        add(WALL_FIXER, 10)

    if (any(role.health < (110 if role.kind == P.WORKER else 100)
            for role in turn.controllable())
            and not any(role.has(P.MEDICINE) for role in turn.controllable())):
        add(P.MEDICINE, 10)

    # 第三天最后 20 回合仍未完成中央双墙三级时，将其提升为
    # 夜战应急采购，突破普通购物在第 60 回合后停止的限制。
    day_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
    levels = sorted((weapon.level for weapon in turn.weapons()), reverse=True)
    fire_anchor = len(levels) >= 3 and levels[0] >= 3
    central_need = central_wall_upgrade_need(turn)
    if (P.day_index(turn.round_no) == 3 and day_round >= 50
            and fire_anchor and central_need is not None):
        name, quantity = central_need
        price = turn.shop_prices.get(name, 20 if name == WALL_UP_V1 else 30)
        carried = sum(role.count(name) for role in turn.controllable())
        missing = max(0, quantity - carried)
        affordable = min(missing, gold // max(1, price))
        if affordable:
            urgent_station = any(item in {STATION_UP_V1, STATION_UP_V2} for item, _ in items)
            if urgent_station:
                items.append((name, affordable))
            else:
                items.insert(0, (name, affordable))
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

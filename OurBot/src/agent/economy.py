"""白天经济: 选矿采集、批量卖矿、建造围墙、商店购物(升级/修复)。

设计要点:
- 工人以"收益 = 价格 / 往返步数"选矿, 并结合新闻推断的停产预期(推理类任务联动)。
- 达到批量阈值或背包将满时去小贩处贩卖(每回合仅一个动作)。
- 金币支出优先级: 基地升级券 > 武器升级券 > 围墙升级/修复; 常备少量石头。
"""
from dataclasses import dataclass, field
from typing import Any

from . import protocol as P
from .grid import next_step, next_step_adjacent, next_step_adjacent_to_any
from .memory import Memory
from .protocol import (
    COPPER,
    GATLING,
    IRON,
    Pos,
    RAILGUN,
    ROCKET,
    STATION,
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

TOWER_LOADOUT = (GATLING, RAILGUN, ROCKET)
WALL_STONE_KEEP = 2  # 工人背包中常备的墙材料石头数
HEALTHY_WALL_RATIO = 0.5  # 墙血量低于最大值的该比例时使用修复包
SELL_BATCH = 8


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
) -> None:
    """单个工人的白天状态机: 建造 > 卖矿 > 采集 > 待命。"""
    day = P.day_index(turn.round_no)

    # 1) 建造优先(武器 > 围墙), 建造是白天限时机会
    if _try_build(turn, worker, plan, claimed, claimed_sites, memory, commands):
        return

    # 2) 矿石达到批量阈值或背包将满时再卖，避免每采一个就横穿地图
    if _try_sell(turn, worker, day, memory, claimed, commands):
        return

    # 3) 背包未满则采集
    if not worker.backpack_full and _try_collect(turn, worker, day, memory, claimed, commands):
        return

    # 4) 没事做: 待在基地附近待命
    _hold_near_station(turn, worker, claimed, commands)


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

    for site, kind in jobs:
        if site in claimed_sites:
            continue
        if distance(worker.pos, site) <= 1:
            commands[worker.unit_id] = build_command(site, kind)
            claimed_sites.add(site)
            return True
        step = next_step(turn, worker, site)
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[worker.unit_id] = move_command(step)
            claimed_sites.add(site)
            return True
    return False


def _try_sell(
    turn: P.Turn,
    worker: Unit,
    day: int,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    vendor = turn.vendor_pos()
    if vendor is None:
        return False
    # 选最值钱的非石头矿石; 石头仅超出常备量才卖
    best_ore, best_num, best_value = None, 0, 0
    for ore in (COPPER, IRON, STONE):
        # 明日停产而今日仍可采时先囤货，等价格上涨后再卖。
        if not memory.ore_blocked(ore, day) and memory.ore_blocked(ore, day + 1):
            continue
        num = worker.count(ore)
        if ore == STONE:
            num = max(0, num - WALL_STONE_KEEP)
        price = turn.vendor_prices.get(ore, 0)
        if num > 0 and num * price > best_value:
            best_ore, best_num, best_value = ore, num, num * price
    if best_ore is None:
        return False
    total_sellable = sum(
        max(0, worker.count(ore) - (WALL_STONE_KEEP if ore == STONE else 0))
        for ore in (COPPER, IRON, STONE)
    )
    nearly_full = worker.capacity is not None and len(worker.backpack) >= int(worker.capacity * 0.8)
    if total_sellable < SELL_BATCH and not nearly_full:
        return False
    if distance(worker.pos, vendor) <= 1:
        commands[worker.unit_id] = sell_command(best_ore, best_num)
        return True
    step = next_step_adjacent(turn, worker, vendor)
    if step is not None and step not in claimed:
        claimed.add(step)
        commands[worker.unit_id] = move_command(step)
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
    for mine in candidates:
        if distance(worker.pos, mine) <= 1:
            claimed.add(mine)
            commands[worker.unit_id] = collect_command(mine)
            return True
        step = next_step_adjacent(turn, worker, mine)
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[worker.unit_id] = move_command(step)
            return True
    return False


def _all_mine_positions(turn: P.Turn) -> list[Pos]:
    return [pos for pos, kind in turn.zones.items() if kind in (STONE, IRON, COPPER)]


def _hold_near_station(
    turn: P.Turn,
    worker: Unit,
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
    step = next_step_adjacent_to_any(turn, worker, list(footprint))
    if step is not None and step not in claimed:
        claimed.add(step)
        commands[worker.unit_id] = move_command(step)


def shopping_list(turn: P.Turn) -> list[tuple[str, int]]:
    """按金币余量生成购物清单(由主决策派最近的持有者去武器商店购买)。"""
    items: list[tuple[str, int]] = []
    gold = turn.gold
    station = turn.station()
    if station is not None and station.level == 1 and gold >= 100:
        items.append((STATION_UP_V1, 1))
        gold -= 100
    if station is not None and station.level == 2 and gold >= 150:
        items.append((STATION_UP_V2, 1))
        gold -= 150
    level1_weapon = next((weapon for weapon in turn.weapons() if weapon.level == 1), None)
    if level1_weapon is not None and gold >= 100:
        items.append((WEAPON_UP_V1, 1))
        gold -= 100
    level2_weapon = next((weapon for weapon in turn.weapons() if weapon.level == 2), None)
    if level2_weapon is not None and gold >= 150:
        items.append((WEAPON_UP_V2, 1))
        gold -= 150
    level1_wall = next((wall for wall in turn.walls() if wall.level == 1), None)
    if level1_wall is not None and gold >= 20:
        items.append((WALL_UP_V1, 1))
        gold -= 20
    level2_wall = next((wall for wall in turn.walls() if wall.level == 2), None)
    if level2_wall is not None and gold >= 30:
        items.append((WALL_UP_V2, 1))
        gold -= 30
    walls = turn.walls()
    if walls and gold >= 10:
        damaged = sum(1 for w in walls if w.health < _wall_max_health(w.level))
        if damaged:
            n = min(damaged, gold // 10)
            items.append((WALL_FIXER, n))
            gold -= 10 * n
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


def _wall_max_health(level: int) -> int:
    return (1000, 1500, 2000)[min(max(level, 1), 3) - 1]

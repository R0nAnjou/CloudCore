"""白天经济: 选矿采集、批量卖矿、建造围墙、商店购物(升级/修复)。

设计要点:
- 工人以"收益 = 价格 / 往返步数"选矿, 并结合新闻推断的停产预期(推理类任务联动)。
- 背包满时优先去小贩处批量贩卖(每回合仅一个动作, 卖矿支持 num 批量)。
- 金币支出优先级: 基地升级券 > 修墙 > 武器升级券; 墙材料常备少量石头。
"""
from dataclasses import dataclass, field
from typing import Any

from . import protocol as P
from .grid import neighbours, next_step
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

    # 2) 背包有矿石则去卖(石头留 WALL_STONE_KEEP 作为墙材料)
    if _try_sell(turn, worker, claimed, commands):
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
        if site not in standing and memory.build_allowed(site, kind):
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
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    vendor = turn.vendor_pos()
    if vendor is None:
        return False
    # 选最值钱的非石头矿石; 石头仅超出常备量才卖
    best_ore, best_num, best_value = None, 0, 0
    for ore in (COPPER, IRON, STONE):
        num = worker.count(ore)
        if ore == STONE:
            num = max(0, num - WALL_STONE_KEEP)
        price = turn.vendor_prices.get(ore, 0)
        if num > 0 and num * price > best_value:
            best_ore, best_num, best_value = ore, num, num * price
    if best_ore is None:
        return False
    if distance(worker.pos, vendor) <= 1:
        commands[worker.unit_id] = sell_command(best_ore, best_num)
        return True
    step = next_step(turn, worker, vendor)
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
    def value(pos: Pos) -> tuple[int, int]:
        ore = turn.zones.get(pos, "")
        price = turn.vendor_prices.get(ore, 0)
        if memory.ore_blocked(ore, day):
            price = 0
        if ore == STONE:
            price = max(price, 2)  # 石头有建造价值, 给保底价
        return (-price, distance(worker.pos, pos))

    candidates = sorted(
        (p for p in _all_mine_positions(turn) if p not in claimed),
        key=value,
    )
    for mine in candidates:
        if distance(worker.pos, mine) <= 1:
            claimed.add(mine)
            commands[worker.unit_id] = collect_command(mine)
            return True
        step = next_step(turn, worker, mine)
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
    for cell in sorted(neighbours(target), key=lambda p: (distance(p, target), p.x, p.y)):
        if turn.land(cell) and cell not in turn.blocked(worker) and cell not in claimed:
            step = next_step(turn, worker, cell)
            if step is not None:
                claimed.add(step)
                commands[worker.unit_id] = move_command(step)
            return


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
    walls = turn.walls()
    if walls and gold >= 10:
        damaged = sum(1 for w in walls if w.health < 1000)
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
        if wall.health < 1000 * HEALTHY_WALL_RATIO and distance(role.pos, wall.pos) <= 1:
            commands[role.unit_id] = use_command(WALL_FIXER, wall.pos)
            return True
    return False

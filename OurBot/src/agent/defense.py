"""夜战指挥官: 目标价值评估、三炮火力分配、弹道校验、集火超杀转移。

武器机制(任务书 4.5.4):
- 加特林: 多目标须在同一 90° 锥形内, 每颗子弹 10 伤害, 命中弹道第一个机器人;
- 电磁狙击: 单目标, 能量沿弹道穿透(可顺带清线);
- 火箭: 落点中心 20 + 周围 8 格溅射 10, 3 回合冷却。
伤害本回合结束统一结算 -> 集火判断按"累计伤害 >= HP"的最小火力集合。
"""
from typing import Any

from . import protocol as P
from .grid import line_cells, neighbours
from .protocol import (
    BOSS_ROBOT,
    GATLING,
    LARGE_ROBOT,
    MIDDLE_ROBOT,
    Pos,
    RAILGUN,
    ROCKET,
    SMALL_ROBOT,
    Robot,
    Unit,
    attack_command,
    distance,
    move_command,
)

ROBOT_THREAT_VALUE = {SMALL_ROBOT: 10, MIDDLE_ROBOT: 9, LARGE_ROBOT: 8, BOSS_ROBOT: 7}


def night(
    turn: P.Turn,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """夜晚主流程: 为每座武器配一名炮手, 选目标开火; 无武器时角色撤回基地。"""
    pairs = _pair_gunners(turn)
    remaining = list(turn.robots)

    # 按武器类型顺序开火: 火箭(群伤) -> 电磁(穿透) -> 加特林(补刀)
    for tower, gunner in sorted(pairs, key=lambda p: (p[0].cooldown > 0, p[0].kind != ROCKET, p[0].kind != RAILGUN)):
        if distance(gunner.pos, tower.pos) > 1:
            # 炮手不在位: 先归位(每回合一步)
            from .grid import next_step

            step = next_step(turn, gunner, tower.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[gunner.unit_id] = move_command(step)
            continue
        if tower.cooldown > 0:
            continue
        targets = _choose_targets(turn, tower, remaining)
        if not targets:
            continue
        commands[tower.unit_id] = attack_command(gunner.unit_id, targets)
        _deduct_expected(turn, tower, targets, remaining)
        claimed.add(gunner.pos)

    # 没有配到武器的角色: 撤回基地附近避险
    _evacuate_idle_roles(turn, pairs, claimed, commands)


def _pair_gunners(turn: P.Turn) -> list[tuple[Unit, Unit]]:
    """角色-武器配对: 就近贪心(角色数 <= 武器数)。"""
    roles = list(turn.controllable())
    towers = list(turn.weapons())
    pairs: list[tuple[Unit, Unit]] = []
    used_roles: set[int] = set()
    for tower in towers:
        best_role, best_d = None, 10**9
        for role in roles:
            if role.unit_id in used_roles:
                continue
            d = distance(role.pos, tower.pos)
            if d < best_d:
                best_role, best_d = role, d
        if best_role is not None:
            used_roles.add(best_role.unit_id)
            pairs.append((tower, best_role))
    return pairs


def _choose_targets(turn: P.Turn, tower: Unit, robots: list[Robot]) -> list[P.Pos]:
    reach = tower.range_of_attack()
    in_range = [r for r in robots if r.health > 0 and not r.dizzy and distance(tower.pos, r.pos) <= reach]
    if not in_range:
        return []

    if tower.kind == ROCKET:
        # 群伤: 选"落点期望伤害/剩余血量溢出"最优的中心
        best_pos, best_value = None, -1
        for center in in_range:
            cluster = [r for r in in_range if distance(center.pos, r.pos) <= 1]
            expected = sum(min(20 if r.pos == center.pos else 10, r.health) for r in cluster)
            value = expected
            if value > best_value:
                best_pos, best_value = center.pos, value
        return [best_pos] if best_pos else []

    if tower.kind == RAILGUN:
        # 穿透: 选弹道上"累计可打伤害"最大的目标
        best_pos, best_value = None, -1
        for robot in in_range:
            path = line_cells(tower.pos, robot.pos)
            blockers = [r for r in robots if r.pos in path and distance(tower.pos, r.pos) <= reach]
            energy = 10 * tower.level
            value = 0
            for r in sorted(blockers, key=lambda r: distance(tower.pos, r.pos)):
                value += min(energy, r.health)
                energy -= min(energy, r.health)
                if energy <= 0:
                    break
            if value > best_value:
                best_pos, best_value = robot.pos, value
        return [best_pos] if best_pos else []

    # 加特林: 目标数 = 等级, 须在同一 90° 锥形内
    max_targets = tower.level
    ordered = sorted(
        in_range,
        key=lambda r: (-ROBOT_THREAT_VALUE.get(r.kind, 5), r.health, distance(tower.pos, r.pos)),
    )
    chosen: list[P.Pos] = []
    for seed in ordered:
        if any(seed.pos == c for c in chosen):
            continue
        group = [seed]
        for other in ordered:
            if other is seed or any(other.pos == c for c in chosen):
                continue
            if len(group) >= max_targets:
                break
            if _within_90deg(tower.pos, seed.pos, other.pos):
                group.append(other)
        if group:
            chosen.extend(r.pos for r in group)
        if len(chosen) >= max_targets:
            break
    return chosen[:max_targets]


def _within_90deg(origin: P.Pos, a: P.Pos, b: P.Pos) -> bool:
    """以 origin 为顶点, a/b 两个方向向量夹角是否 <= 90°。"""
    ax, ay = a.x - origin.x, a.y - origin.y
    bx, by = b.x - origin.x, b.y - origin.y
    dot = ax * bx + ay * by
    return dot >= 0  # 夹角 <= 90° 等价于点积 >= 0(格点向量)


def _deduct_expected(
    turn: P.Turn,
    tower: Unit,
    targets: list[P.Pos],
    remaining: list[Robot],
) -> None:
    """预扣本回合伤害, 让后续武器的目标选择知道'这只怪会被打掉多少血'。"""
    if tower.kind == ROCKET:
        for center in targets:
            for r in remaining:
                if r.pos == center:
                    r.health -= 20
                elif distance(center, r.pos) <= 1:
                    r.health -= 10
    elif tower.kind == RAILGUN:
        energy = 10 * tower.level
        path = line_cells(tower.pos, targets[0]) if targets else []
        for r in sorted(remaining, key=lambda r: distance(tower.pos, r.pos)):
            if r.pos in path:
                dmg = min(energy, r.health)
                r.health -= dmg
                energy -= dmg
                if energy <= 0:
                    break
    else:  # gatling
        for pos in targets:
            path = line_cells(tower.pos, pos)
            for r in sorted(remaining, key=lambda r: distance(tower.pos, r.pos)):
                if r.pos in path:
                    r.health -= 10
                    break


def _evacuate_idle_roles(
    turn: P.Turn,
    pairs: list[tuple[Unit, Unit]],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    from .grid import next_step
    from .protocol import station_footprint

    assigned = {g.unit_id for _, g in pairs}
    station = turn.station()
    if station is None:
        return
    footprint = station_footprint(station.pos)
    shelter = footprint[2]  # 基地左下格
    for role in turn.controllable():
        if role.unit_id in assigned or role.unit_id in commands:
            continue
        if distance(role.pos, shelter) <= 1:
            continue
        step = next_step(turn, role, shelter)
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[role.unit_id] = move_command(step)

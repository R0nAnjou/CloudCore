"""夜战指挥官: 目标价值评估、三炮火力分配、弹道校验、集火超杀转移。

武器机制(任务书 4.5.4):
- 加特林: 多目标须在同一 90° 锥形内, 每颗子弹 10 伤害, 命中弹道第一个机器人;
- 电磁狙击: 单目标, 能量沿弹道穿透(可顺带清线);
- 火箭: 落点中心 20 + 周围 8 格溅射 10, 3 回合冷却。
伤害本回合结束统一结算 -> 集火判断按"累计伤害 >= HP"的最小火力集合。
"""
from itertools import combinations, permutations
from typing import Any

from . import protocol as P
from .grid import line_cells, neighbours, next_step_adjacent, next_step_adjacent_to_any
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

ROBOT_THREAT_VALUE = {SMALL_ROBOT: 5, MIDDLE_ROBOT: 12, LARGE_ROBOT: 30, BOSS_ROBOT: 60}


def night(
    turn: P.Turn,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    unavailable_role_ids: set[int] | None = None,
) -> None:
    """夜晚主流程: 为每座武器配一名炮手, 选目标开火; 无武器时角色撤回基地。"""
    unavailable_role_ids = unavailable_role_ids or set()
    pairs = _pair_gunners(turn, unavailable_role_ids)
    remaining_hp = {robot.robot_id: robot.health for robot in turn.robots}

    # 按武器类型顺序开火: 火箭(群伤) -> 电磁(穿透) -> 加特林(补刀)
    for tower, gunner in sorted(pairs, key=lambda p: (p[0].cooldown > 0, p[0].kind != ROCKET, p[0].kind != RAILGUN)):
        if distance(gunner.pos, tower.pos) > 1:
            # 炮手不在位: 先归位(每回合一步)
            step = next_step_adjacent(turn, gunner, tower.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[gunner.unit_id] = move_command(step)
            continue
        if tower.cooldown > 0:
            continue
        targets = _choose_targets(turn, tower, list(turn.robots), remaining_hp)
        if not targets:
            continue
        commands[tower.unit_id] = attack_command(gunner.unit_id, targets)
        _deduct_expected(tower, targets, list(turn.robots), remaining_hp)
        claimed.add(gunner.pos)

    # 没有配到武器的角色: 撤回基地附近避险
    _evacuate_idle_roles(turn, pairs, claimed, commands, unavailable_role_ids)


def _pair_gunners(
    turn: P.Turn,
    unavailable_role_ids: set[int] | None = None,
) -> list[tuple[Unit, Unit]]:
    """最多三人三炮，穷举配对以优先保证本回合能开火。"""
    unavailable_role_ids = unavailable_role_ids or set()
    roles = [r for r in turn.controllable() if r.unit_id not in unavailable_role_ids]
    towers = list(turn.weapons())
    n = min(len(roles), len(towers))
    if n == 0:
        return []
    priority = {ROCKET: 3, RAILGUN: 2, GATLING: 1}
    best_pairs: list[tuple[Unit, Unit]] = []
    best_score = -10**9
    for selected_towers in combinations(towers, n):
        for selected_roles in permutations(roles, n):
            pairs = list(zip(selected_towers, selected_roles))
            score = sum(
                (100 + 5 * priority.get(tower.kind, 0)) if distance(tower.pos, role.pos) <= 1
                else (2 * priority.get(tower.kind, 0) - 10 * distance(tower.pos, role.pos))
                for tower, role in pairs
            )
            if score > best_score:
                best_pairs, best_score = pairs, score
    return best_pairs


def _choose_targets(
    turn: P.Turn,
    tower: Unit,
    robots: list[Robot],
    remaining_hp: dict[int, int] | None = None,
) -> list[P.Pos]:
    remaining_hp = remaining_hp or {robot.robot_id: robot.health for robot in robots}
    reach = tower.range_of_attack()
    in_range = [
        r for r in robots
        if remaining_hp.get(r.robot_id, 0) > 0 and distance(tower.pos, r.pos) <= reach
    ]
    if not in_range:
        return []

    if tower.kind == ROCKET:
        # 导弹允许落点重叠；逐枚选择边际收益最大的格，确保数量始终等于等级。
        candidates = {
            pos
            for robot in in_range
            for pos in (robot.pos, *neighbours(robot.pos))
            if 0 <= pos.x < turn.width
            and 0 <= pos.y < turn.height
            and distance(tower.pos, pos) <= reach
        }
        simulated = dict(remaining_hp)
        chosen: list[Pos] = []
        for _ in range(max(1, tower.level)):
            best = max(
                candidates,
                key=lambda center: _rocket_value(turn, center, in_range, simulated),
            )
            chosen.append(best)
            _apply_rocket(best, in_range, simulated)
        return chosen

    if tower.kind == RAILGUN:
        # 穿透: 选弹道上"累计可打伤害"最大的目标
        best_pos, best_value = None, -1
        for robot in in_range:
            path = line_cells(tower.pos, robot.pos)
            blockers = [
                r for r in robots
                if remaining_hp.get(r.robot_id, 0) > 0
                and r.pos in path
                and distance(tower.pos, r.pos) <= reach
            ]
            energy = 10 * tower.level
            value = 0
            for r in sorted(blockers, key=lambda r: distance(tower.pos, r.pos)):
                hp = remaining_hp[r.robot_id]
                value += min(energy, hp)
                energy -= min(energy, hp)
                if energy <= 0:
                    break
            if value > best_value:
                best_pos, best_value = robot.pos, value
        return [best_pos] if best_pos else []

    # 加特林: 目标数 = 等级, 须在同一 90° 锥形内
    max_targets = tower.level
    ordered = sorted(in_range, key=lambda r: _robot_priority(turn, tower, r, remaining_hp))
    best_group: list[Robot] = []
    best_score = -1
    for seed in ordered:
        group = [seed]
        for other in ordered:
            if other.robot_id == seed.robot_id or len(group) >= max_targets:
                continue
            if all(_within_90deg(tower.pos, member.pos, other.pos) for member in group):
                group.append(other)
        score = sum(_robot_value(turn, r, remaining_hp) for r in group)
        if score > best_score:
            best_group, best_score = group, score
    chosen = [robot.pos for robot in best_group[:max_targets]]
    # 规则要求目标数与等级一致。重复同一路径表示多颗子弹沿同方向射击。
    while chosen and len(chosen) < max_targets:
        chosen.append(chosen[0])
    return chosen


def _robot_value(turn: P.Turn, robot: Robot, remaining_hp: dict[int, int]) -> int:
    targets_us = not robot.target_team or robot.target_team == turn.team_type
    protection_bonus = 100 if targets_us else 0
    return protection_bonus + ROBOT_THREAT_VALUE.get(robot.kind, 5) + robot.score * 4 - min(
        remaining_hp.get(robot.robot_id, robot.health), 100
    ) // 20


def _robot_priority(
    turn: P.Turn,
    tower: Unit,
    robot: Robot,
    remaining_hp: dict[int, int],
) -> tuple[int, int, int]:
    return (
        -_robot_value(turn, robot, remaining_hp),
        remaining_hp.get(robot.robot_id, robot.health),
        distance(tower.pos, robot.pos),
    )


def _rocket_value(
    turn: P.Turn,
    center: Pos,
    robots: list[Robot],
    remaining_hp: dict[int, int],
) -> int:
    value = 0
    for robot in robots:
        hp = remaining_hp.get(robot.robot_id, 0)
        if hp <= 0 or distance(center, robot.pos) > 1:
            continue
        damage = 20 if robot.pos == center else 10
        multiplier = 3 if not robot.target_team or robot.target_team == turn.team_type else 1
        value += min(damage, hp) * multiplier + robot.score
    return value


def _apply_rocket(center: Pos, robots: list[Robot], remaining_hp: dict[int, int]) -> None:
    for robot in robots:
        hp = remaining_hp.get(robot.robot_id, 0)
        if hp <= 0 or distance(center, robot.pos) > 1:
            continue
        remaining_hp[robot.robot_id] = hp - (20 if robot.pos == center else 10)


def _within_90deg(origin: P.Pos, a: P.Pos, b: P.Pos) -> bool:
    """以 origin 为顶点, a/b 两个方向向量夹角是否 <= 90°。"""
    ax, ay = a.x - origin.x, a.y - origin.y
    bx, by = b.x - origin.x, b.y - origin.y
    dot = ax * bx + ay * by
    return dot >= 0  # 夹角 <= 90° 等价于点积 >= 0(格点向量)


def _deduct_expected(
    tower: Unit,
    targets: list[P.Pos],
    robots: list[Robot],
    remaining_hp: dict[int, int],
) -> None:
    """预扣本回合伤害, 让后续武器的目标选择知道'这只怪会被打掉多少血'。"""
    if tower.kind == ROCKET:
        for center in targets:
            _apply_rocket(center, robots, remaining_hp)
    elif tower.kind == RAILGUN:
        energy = 10 * tower.level
        path = line_cells(tower.pos, targets[0]) if targets else []
        for r in sorted(robots, key=lambda r: distance(tower.pos, r.pos)):
            hp = remaining_hp.get(r.robot_id, 0)
            if hp > 0 and r.pos in path:
                dmg = min(energy, hp)
                remaining_hp[r.robot_id] = hp - dmg
                energy -= dmg
                if energy <= 0:
                    break
    else:  # gatling
        for pos in targets:
            path = line_cells(tower.pos, pos)
            for r in sorted(robots, key=lambda r: distance(tower.pos, r.pos)):
                hp = remaining_hp.get(r.robot_id, 0)
                if hp > 0 and r.pos in path:
                    remaining_hp[r.robot_id] = hp - 10
                    break


def _evacuate_idle_roles(
    turn: P.Turn,
    pairs: list[tuple[Unit, Unit]],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    unavailable_role_ids: set[int] | None = None,
) -> None:
    from .protocol import station_footprint

    assigned = {g.unit_id for _, g in pairs}
    unavailable_role_ids = unavailable_role_ids or set()
    station = turn.station()
    if station is None:
        return
    footprint = station_footprint(station.pos)
    shelter = footprint[2]  # 基地左下格
    for role in turn.controllable():
        if (role.unit_id in assigned or role.unit_id in commands
                or role.unit_id in unavailable_role_ids):
            continue
        if distance(role.pos, shelter) <= 1:
            continue
        step = next_step_adjacent_to_any(turn, role, list(footprint))
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[role.unit_id] = move_command(step)

"""寻路与几何: 8 向 A*(切比雪夫启发) + 直线弹道采样。"""
from heapq import heappop, heappush
from itertools import count

from .protocol import Pos, Turn, Unit, distance

STEPS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def next_step(turn: Turn, moving: Unit, goal: Pos) -> Pos | None:
    """返回朝 goal 的下一格; 不可达返回 None。每次调用重算(地图小, 开销可接受)。"""
    return next_step_to_any(turn, moving, (goal,))


def next_step_adjacent(turn: Turn, moving: Unit, target: Pos, *, reserved: set[Pos] | None = None) -> Pos | None:
    """走向交互目标周围一格，而不是尝试走进矿点/商店/建筑本身。"""
    return next_step_adjacent_to_any(turn, moving, (target,), reserved=reserved)


def next_step_adjacent_to_any(
    turn: Turn,
    moving: Unit,
    targets: tuple[Pos, ...] | list[Pos],
    *,
    reserved: set[Pos] | None = None,
) -> Pos | None:
    stands = adjacent_stands(turn, moving, targets)
    return next_step_to_any(turn, moving, stands, reserved=reserved)


def adjacent_stands(turn: Turn, moving: Unit, targets) -> set[Pos]:
    blocked = turn.blocked(moving)
    stands = {
        Pos(target.x + dx, target.y + dy)
        for target in targets
        for dx, dy in STEPS
    }
    stands = {
        pos
        for pos in stands
        if turn.land(pos) and (pos == moving.pos or pos not in blocked)
    }
    return stands


def next_step_to_any(
    turn: Turn,
    moving: Unit,
    goals: tuple[Pos, ...] | list[Pos] | set[Pos],
    *,
    reserved: set[Pos] | None = None,
) -> Pos | None:
    """返回到一组可站立终点中最近一个的下一步。"""
    path = path_to_any(turn, moving, goals, reserved=reserved)
    return path[0] if path else None


def path_to_any(turn: Turn, moving: Unit, goals, *, reserved: set[Pos] | None = None) -> list[Pos] | None:
    """完整路线用于返城预算；空列表表示已经到达，None 表示暂时不可达。"""
    goals = {
        goal
        for goal in goals
        if turn.land(goal)
    }
    if not goals:
        return None
    if moving.pos in goals:
        return []
    blocked = turn.blocked(moving) | (reserved or set())
    goals = {goal for goal in goals if goal == moving.pos or goal not in blocked}
    if not goals:
        return None

    def heuristic(pos: Pos) -> int:
        return min(distance(pos, goal) for goal in goals)

    order = count()
    frontier: list[tuple[int, int, int, Pos]] = [
        (heuristic(moving.pos), 0, next(order), moving.pos)
    ]
    came_from: dict[Pos, Pos] = {}
    best: dict[Pos, int] = {moving.pos: 0}
    seen: set[Pos] = set()

    while frontier:
        _, cost, _, current = heappop(frontier)
        if current in seen:
            continue
        if current in goals:
            path = []
            while current != moving.pos:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))
        seen.add(current)
        for dx, dy in STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in blocked or not turn.land(step):
                continue
            new_cost = cost + 1
            if new_cost >= best.get(step, new_cost + 1):
                continue
            best[step] = new_cost
            came_from[step] = current
            heappush(frontier, (new_cost + heuristic(step), new_cost, next(order), step))
    return None


def _first_step(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Pos:
    current = goal
    while came_from[current] != start:
        current = came_from[current]
    return current


def line_cells(a: Pos, b: Pos) -> list[Pos]:
    """两点间直线弹道经过的格子(Bresenham, 含两端)。"""
    cells: list[Pos] = []
    x0, y0, x1, y1 = a.x, a.y, b.x, b.y
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append(Pos(x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def neighbours(pos: Pos) -> tuple[Pos, ...]:
    return tuple(Pos(pos.x + dx, pos.y + dy) for dx, dy in STEPS)

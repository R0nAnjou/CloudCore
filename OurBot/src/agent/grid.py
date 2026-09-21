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
    blocked = turn.blocked(moving)
    order = count()
    frontier: list[tuple[int, int, int, Pos]] = [
        (distance(moving.pos, goal), 0, next(order), moving.pos)
    ]
    came_from: dict[Pos, Pos] = {}
    best: dict[Pos, int] = {moving.pos: 0}
    seen: set[Pos] = set()

    while frontier:
        _, cost, _, current = heappop(frontier)
        if current in seen:
            continue
        if current == goal:
            return _first_step(came_from, moving.pos, goal)
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
            heappush(frontier, (new_cost + distance(step, goal), new_cost, next(order), step))
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

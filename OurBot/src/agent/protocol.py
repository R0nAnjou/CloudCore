"""协议层: 请求 JSON -> 内部模型, 以及全部动作码的指令构造器。

坐标约定: 原点(0,0)在左下角, x 向右, y 向上。
距离约定: 切比雪夫距离 max(|dx|, |dy|)。
"""
from dataclasses import dataclass
from typing import Any

# ---- 时间常量 ----------------------------------------------------------
DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS  # 130
TOTAL_DAYS = 10

# ---- 物品/单位常量 -----------------------------------------------------
STATION = "station"
WALL = "wall"
WORKER = "worker"
PIONEER = "pioneer"
GATLING = "gatling"
RAILGUN = "railgun"
ROCKET = "rocket"
TOWER_TYPES = (GATLING, RAILGUN, ROCKET)
CONTROLLABLE_TYPES = (WORKER, PIONEER)
WEAPON_BUILD_COST = 25
WALL_MATERIAL = "stone"
LAND = "land"

# 矿石类型(与 zones.neutralType / backpack 名称一致)
STONE = "stone"
IRON = "iron"
COPPER = "copper"
ORES = (STONE, IRON, COPPER)

# 武器商店物品
WEAPON_UP_V1 = "WeaponUpgradeVoucher1"
WEAPON_UP_V2 = "WeaponUpgradeVoucher2"
WALL_UP_V1 = "WallUpgradeVoucher1"
WALL_UP_V2 = "WallUpgradeVoucher2"
STATION_UP_V1 = "StationUpgradeVoucher1"
STATION_UP_V2 = "StationUpgradeVoucher2"
WALL_FIXER = "WallFixer"
MEDICINE = "Medicine"
DIZZY = "DizzyWeapon"
BOMB = "Bomb"
SMALL_ORDER = "SmallRobotSummonOrder"
MIDDLE_ORDER = "MiddleRobotSummonOrder"
LARGE_ORDER = "LargeRobotSummonOrder"
BOSS_ORDER = "BossRobotSummonOrder"

# 机器人类型
SMALL_ROBOT = "smallRobot"
MIDDLE_ROBOT = "middleRobot"
LARGE_ROBOT = "largeRobot"
BOSS_ROBOT = "bossRobot"

# 机器人积分与 HP(任务书 4.7.2)
ROBOT_SCORE = {SMALL_ROBOT: 1, MIDDLE_ROBOT: 2, LARGE_ROBOT: 4, BOSS_ROBOT: 10}
ROBOT_MAX_HP = {SMALL_ROBOT: 40, MIDDLE_ROBOT: 60, LARGE_ROBOT: 500, BOSS_ROBOT: 800}

TOWER_RANGE_BY_LEVEL = {
    GATLING: (3, 5, 7),
    RAILGUN: (6, 8, 10),
    ROCKET: (10, 15, 10**9),
}


def is_day_round(round_no: int) -> bool:
    """白天=每游戏日前70回合。"""
    return (round_no - 1) % ROUNDS_PER_DAY < DAY_ROUNDS


def day_index(round_no: int) -> int:
    """当前第几天(1..10)。"""
    return (round_no - 1) // ROUNDS_PER_DAY + 1


# ---- 基础结构 ----------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Pos:
    x: int
    y: int

    @classmethod
    def load(cls, raw: Any) -> "Pos":
        return cls(int(raw["x"]), int(raw["y"]))

    def dump(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


def distance(first: Pos, second: Pos) -> int:
    return max(abs(first.x - second.x), abs(first.y - second.y))


def station_footprint(pos: Pos) -> tuple[Pos, ...]:
    """基地 2x2, pos 传的是左上角坐标(接口文档 1.3.2)。"""
    return (
        pos,
        Pos(pos.x + 1, pos.y),
        Pos(pos.x, pos.y - 1),
        Pos(pos.x + 1, pos.y - 1),
    )


@dataclass(frozen=True, slots=True)
class Unit:
    unit_id: int
    pos: Pos
    kind: str
    health: int
    level: int
    cooldown: int
    attack_range: int
    capacity: int | None
    backpack: tuple[str, ...]

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Unit":
        raw_capacity = raw.get("backPackCapability")
        return cls(
            int(raw.get("id") or 0),
            Pos.load(raw["pos"]),
            str(raw["roleType"]),
            int(raw["health"]),
            int(raw.get("level") or 0),
            int(raw.get("cooldown") or 0),
            int(raw.get("attackRange") or 0),
            int(raw_capacity) if raw_capacity is not None else None,
            tuple(str(item) for item in raw.get("backpack") or ()),
        )

    def has(self, item: str) -> bool:
        return item in self.backpack

    def count(self, item: str) -> int:
        return self.backpack.count(item)

    @property
    def backpack_full(self) -> bool:
        if self.capacity is None:
            return False
        return len(self.backpack) >= self.capacity

    def range_of_attack(self) -> int:
        if self.attack_range > 0:
            return self.attack_range
        table = TOWER_RANGE_BY_LEVEL.get(self.kind)
        if table is None:
            return 0
        level = min(max(self.level, 1), len(table))
        return table[level - 1]


@dataclass(frozen=True, slots=True)
class Robot:
    robot_id: int
    pos: Pos
    kind: str
    health: int
    dizzy: bool
    target_team: str

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Robot":
        return cls(
            int(raw["id"]),
            Pos.load(raw["pos"]),
            str(raw["roleType"]),
            int(raw["health"]),
            str(raw.get("abnormalState") or "") == "dizzy",
            str(raw.get("targetTeam") or ""),
        )

    @property
    def score(self) -> int:
        return ROBOT_SCORE.get(self.kind, 1)


@dataclass(frozen=True, slots=True)
class PlayerTask:
    task_type: str
    position: Pos
    cold_down: int
    score_reward: int
    gold_reward: int
    is_valid: bool
    timeout_rounds: int

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "PlayerTask":
        return cls(
            str(raw.get("taskType") or ""),
            Pos.load(raw["taskPosition"]),
            int(raw.get("coldDownRounds") or 0),
            int(raw.get("scoreReward") or 0),
            int(raw.get("goldReward") or 0),
            bool(raw.get("isValid")),
            int(raw.get("timeoutRounds") or 0),
        )


@dataclass(frozen=True, slots=True)
class ShopItem:
    name: str
    price: int

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "ShopItem":
        return cls(str(raw.get("name") or ""), int(raw.get("price") or 0))


@dataclass(frozen=True, slots=True)
class Turn:
    """单回合状态快照(仅包含决策所需字段)。"""

    round_no: int
    is_day: bool
    gold: int
    width: int
    height: int
    zones: dict[Pos, str]
    ours: tuple[Unit, ...]
    enemy_roles: tuple[Unit, ...]
    robots: tuple[Robot, ...]
    player_tasks: tuple[PlayerTask, ...]
    phase_task: str
    llm_resp: str
    last_cmd_result: str
    world_official_news: str
    world_folk_legends: str
    last_action_results: dict[int, bool]
    last_summon_result: int
    vendor_prices: dict[str, int]
    shop_prices: dict[str, int]

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "Turn":
        round_no = int(payload["roundNo"])
        info = payload["mapInfo"]
        team = payload["teamOur"]
        world = payload.get("worldNews") or {}
        return cls(
            round_no,
            is_day_round(round_no),
            int(team.get("goldNum") or 0),
            int(info["width"]),
            int(info["height"]),
            {
                Pos.load(zone["pos"]): str(zone["neutralType"])
                for zone in info.get("zones") or ()
            },
            tuple(Unit.load(role) for role in team.get("roles") or ()),
            tuple(Unit.load(role) for role in (payload.get("teamEnemy") or {}).get("roles") or ()),
            tuple(
                Robot.load(robot)
                for robot in (payload.get("robot") or {}).get("roles") or ()
            ),
            tuple(PlayerTask.load(t) for t in team.get("playerTasks") or ()),
            str(payload.get("phaseTask") or ""),
            str(payload.get("llmResp") or ""),
            str(payload.get("lastCmdResult") or ""),
            str(world.get("officialNews") or ""),
            str(world.get("folkLegends") or ""),
            {
                int(k): bool(v)
                for k, v in (payload.get("lastRoundRoleActionResults") or {}).items()
            },
            int(payload.get("lastSummonTreasureResult") or 0),
            {
                str(v.get("name")): int(v.get("price") or 0)
                for v in payload.get("vendorShopList") or ()
            },
            {
                str(v.get("name")): int(v.get("price") or 0)
                for v in payload.get("weaponShopList") or ()
            },
        )

    # ---- 单位查询 ------------------------------------------------------
    def station(self) -> Unit | None:
        for unit in self.ours:
            if unit.kind == STATION:
                return unit
        return None

    def alive(self, kinds: tuple[str, ...]) -> tuple[Unit, ...]:
        return tuple(u for u in self.ours if u.kind in kinds and u.health > 0)

    def controllable(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive(CONTROLLABLE_TYPES), key=lambda u: u.unit_id))

    def workers(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive((WORKER,)), key=lambda u: u.unit_id))

    def pioneers(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive((PIONEER,)), key=lambda u: u.unit_id))

    def weapons(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive(TOWER_TYPES), key=lambda u: u.unit_id))

    def walls(self) -> tuple[Unit, ...]:
        return self.alive((WALL,))

    def foot_print(self, unit: Unit) -> tuple[Pos, ...]:
        if unit.kind == STATION:
            return station_footprint(unit.pos)
        return (unit.pos,)

    # ---- 地形查询 ------------------------------------------------------
    def neutral_cells(self) -> frozenset[Pos]:
        return frozenset(self.zones)

    def land(self, pos: Pos) -> bool:
        """可通行/可站立空地。"""
        if not 0 <= pos.x < self.width or not 0 <= pos.y < self.height:
            return False
        return self.zones.get(pos, LAND) == LAND

    def buildable_weapon(self, pos: Pos) -> bool:
        return self.land(pos)

    def mines(self, ore: str | None = None) -> tuple[Pos, ...]:
        return tuple(
            pos
            for pos, kind in self.zones.items()
            if kind == (ore or True) or (ore is not None and kind == ore)
        ) if ore else tuple(pos for pos, k in self.zones.items() if k in ORES)

    def mines_of(self, ore: str) -> tuple[Pos, ...]:
        return tuple(pos for pos, kind in self.zones.items() if kind == ore)

    def vendor_pos(self) -> Pos | None:
        for pos, kind in self.zones.items():
            if kind == "vendor":
                return pos
        return None

    def weapon_shop_pos(self) -> Pos | None:
        for pos, kind in self.zones.items():
            if kind == "weaponShop":
                return pos
        return None

    def task_points(self) -> tuple[Pos, ...]:
        return tuple(
            pos
            for pos, kind in self.zones.items()
            if kind.startswith("challengerTask") or kind.startswith("defenderTask")
        )

    # ---- 障碍与寻路辅助 ------------------------------------------------
    def occupied_cells(self) -> set[Pos]:
        cells: set[Pos] = set()
        for unit in self.ours:
            cells.update(self.foot_print(unit))
        return cells

    def blocked(self, moving: Unit) -> set[Pos]:
        cells = {pos for pos, kind in self.zones.items() if kind != LAND}
        cells.update(self.occupied_cells())
        cells.discard(moving.pos)
        for robot in self.robots:
            cells.add(robot.pos)
        return cells


# ---- 指令构造器(动作码全集) -------------------------------------------
def move_command(pos: Pos) -> dict[str, Any]:
    return {"action": "move", "targetPos": [pos.dump()]}


def collect_command(pos: Pos) -> dict[str, Any]:
    return {"action": "collect", "targetPos": [pos.dump()]}


def build_command(pos: Pos, name: str) -> dict[str, Any]:
    return {"action": "build", "targetPos": [pos.dump()], "name": name}


def remove_command(pos: Pos) -> dict[str, Any]:
    return {"action": "remove", "targetPos": [pos.dump()]}


def sell_command(name: str, num: int = 1) -> dict[str, Any]:
    return {"action": "sell", "name": name, "num": int(num)}


def buy_command(name: str, num: int = 1) -> dict[str, Any]:
    return {"action": "buy", "name": name, "num": int(num)}


def use_command(name: str, pos: Pos | None = None) -> dict[str, Any]:
    cmd: dict[str, Any] = {"action": "use", "name": name}
    if pos is not None:
        cmd["targetPos"] = [pos.dump()]
    return cmd


def attack_command(controller_id: int, targets: list[Pos]) -> dict[str, Any]:
    """操控武器攻击。targets 数量须与武器当前等级一致(任务书 4.5.4)。"""
    return {
        "action": "attack",
        "controllerId": str(controller_id),
        "targetPos": [p.dump() for p in targets],
    }


def accept_task_command() -> dict[str, Any]:
    return {"action": "acceptTask"}


def submit_answer_command(answer: str) -> dict[str, Any]:
    return {"action": "submitAnswer", "taskAnswer": answer}


def summon_treasure_command(pos: Pos, items: list[str]) -> dict[str, Any]:
    return {"action": "summonTreasure", "targetPos": [pos.dump()], "item": list(items)}


def drop_command(name: str) -> dict[str, Any]:
    return {"action": "drop", "name": name}

"""Regression tests for first-night construction and wall repair."""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import brain, defense, economy  # noqa: E402
from agent.memory import Memory  # noqa: E402
from agent.protocol import Pos, Robot, STATION_UP_V1, Turn, WALL_FIXER  # noqa: E402


class DefenseEconomyTests(unittest.TestCase):
    def setUp(self) -> None:
        sample = json.loads((ROOT.parent / "docs" / "request.txt").read_text(encoding="utf-8"))
        self.turn = Turn.load(sample)
        self.station = self.turn.station()
        self.worker = self.turn.workers()[0]
        self.wall = self.turn.walls()[0]
        self.assertIsNotNone(self.station)

    def test_wall_gate_closes_only_after_workers_are_inside(self) -> None:
        turn = replace(self.turn, round_no=1, is_day=True, ours=(self.station,), zones={})
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(turn)
        self.assertEqual(3, len(plan.tower_sites))
        self.assertEqual(("rocket", "rocket", "rocket"), plan.tower_kinds)
        self.assertEqual(19, len(plan.wall_sites))
        inside_worker = replace(self.worker, pos=Pos(9, 24))
        turn = replace(turn, round_no=65, ours=(self.station, inside_worker))
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(turn)
        self.assertEqual(20, len(plan.wall_sites))
        outside_worker = replace(inside_worker, pos=Pos(5, 22))
        turn = replace(turn, ours=(self.station, outside_worker))
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(turn)
        self.assertEqual(19, len(plan.wall_sites))

    def test_outside_income_worker_does_not_block_gate_closing(self) -> None:
        defender = replace(self.turn.workers()[0], pos=Pos(9, 24), backpack=("stone",))
        miner = replace(self.turn.workers()[1], pos=Pos(5, 22), backpack=())
        turn = replace(
            self.turn, round_no=65, is_day=True, zones={},
            ours=(self.station, defender, miner), player_tasks=(), phase_task="",
        )
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(turn)
            self.assertEqual(20, len(plan.wall_sites))
            self.assertNotIn(miner.unit_id, brain.MEMORY.returning_roles)

    def test_worker_returns_inside_before_first_night(self) -> None:
        worker = replace(self.worker, pos=Pos(5, 22), backpack=("stone",))
        turn = replace(
            self.turn, round_no=60, is_day=True, gold=0, zones={},
            ours=(self.station, worker), player_tasks=(), phase_task="",
        )
        with patch.object(brain, "MEMORY", Memory()):
            commands: dict = {}
            brain._day_phase(turn, commands)
        self.assertEqual("move", commands[worker.unit_id]["action"])

    def test_worker_can_enter_nearly_closed_wall_ring(self) -> None:
        empty = replace(self.turn, round_no=1, is_day=True, gold=0, zones={},
                        ours=(self.station,), player_tasks=(), phase_task="")
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(empty)
        gate = next(pos for pos in brain._ring(
            brain.station_footprint(self.station.pos), 2,
        ) if pos not in plan.wall_sites)
        walls = tuple(
            replace(self.wall, unit_id=40000 + index, pos=pos, health=1000)
            for index, pos in enumerate(plan.wall_sites)
        )
        dx = -1 if gate.x < self.station.pos.x else 1
        dy = 1 if gate.y > self.station.pos.y else -1
        worker = replace(self.worker, pos=Pos(gate.x + dx, gate.y + dy), backpack=())
        with patch.object(brain, "MEMORY", Memory()):
            for round_no in range(60, 70):
                turn = replace(empty, round_no=round_no, ours=(self.station, *walls, worker))
                if brain._inside_defense(turn, worker):
                    break
                commands: dict = {}
                brain._day_phase(turn, commands)
                self.assertEqual("move", commands[worker.unit_id]["action"])
                worker = replace(worker, pos=Pos(**commands[worker.unit_id]["targetPos"][0]))
            self.assertTrue(brain._inside_defense(turn, worker))

    def test_returned_worker_closes_last_gate(self) -> None:
        empty = replace(self.turn, round_no=1, is_day=True, gold=0, zones={},
                        ours=(self.station,), player_tasks=(), phase_task="")
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(empty)
        gate = next(pos for pos in brain._ring(
            brain.station_footprint(self.station.pos), 2,
        ) if pos not in plan.wall_sites)
        walls = tuple(
            replace(self.wall, unit_id=40000 + index, pos=pos, health=1000)
            for index, pos in enumerate(plan.wall_sites)
        )
        inner = next(
            pos for pos in brain._ring(brain.station_footprint(self.station.pos), 1)
            if max(abs(pos.x - gate.x), abs(pos.y - gate.y)) == 1
        )
        worker = replace(self.worker, pos=inner, backpack=("stone",))
        turn = replace(empty, round_no=65, ours=(self.station, *walls, worker))
        with patch.object(brain, "MEMORY", Memory()):
            commands: dict = {}
            brain._day_phase(turn, commands)
        self.assertEqual("build", commands[worker.unit_id]["action"])
        self.assertEqual("wall", commands[worker.unit_id]["name"])
        self.assertEqual(gate.dump(), commands[worker.unit_id]["targetPos"][0])

    def test_two_gunners_prioritize_ready_rocket_and_railgun(self) -> None:
        towers = tuple(
            replace(tower, pos=position)
            for tower, position in zip(
                self.turn.weapons(), (Pos(10, 11), Pos(11, 10), Pos(10, 10))
            )
        )
        workers = (
            replace(self.turn.workers()[0], pos=Pos(9, 10)),
            replace(self.turn.workers()[1], pos=Pos(11, 11)),
        )
        turn = replace(self.turn, ours=(*towers, *workers))
        kinds = {tower.kind for tower, _ in defense._pair_gunners(turn)}
        self.assertEqual({"rocket", "railgun"}, kinds)

    def test_existing_three_towers_are_not_rebuilt(self) -> None:
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(self.turn)
        self.assertEqual((), plan.tower_sites)

    def test_worker_collects_stone_in_batches_for_wall(self) -> None:
        worker = replace(self.worker, pos=Pos(5, 22), backpack=())
        turn = replace(
            self.turn, round_no=2, is_day=True, gold=0, zones={Pos(5, 23): "stone"},
            ours=(self.station, worker),
        )
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(turn)
        commands: dict = {}
        economy.worker_day(turn, worker, plan, set(), set(), Memory(), commands)
        self.assertEqual("collect", commands[worker.unit_id]["action"])
        worker = replace(worker, backpack=("stone",) * economy.STONE_BUILD_BATCH)
        turn = replace(turn, ours=(self.station, worker))
        commands = {}
        economy.worker_day(turn, worker, plan, set(), set(), Memory(), commands)
        self.assertIn(commands[worker.unit_id]["action"], {"move", "build"})

    def test_income_worker_chooses_copper_instead_of_near_stone(self) -> None:
        worker = replace(self.worker, pos=Pos(5, 22), backpack=())
        turn = replace(
            self.turn, round_no=2, is_day=True, gold=0,
            zones={Pos(5, 23): "stone", Pos(6, 22): "copper"},
            ours=(self.station, worker),
        )
        commands: dict = {}
        economy.worker_day(
            turn, worker, economy.Plan(), set(), set(), Memory(), commands,
            income_only=True,
        )
        self.assertEqual("collect", commands[worker.unit_id]["action"])
        self.assertEqual({"x": 6, "y": 22}, commands[worker.unit_id]["targetPos"][0])

    def test_income_worker_sells_copper_batch(self) -> None:
        vendor = self.turn.vendor_pos()
        self.assertIsNotNone(vendor)
        worker = replace(
            self.worker, pos=Pos(vendor.x + 1, vendor.y), backpack=("copper",) * 8,
        )
        turn = replace(
            self.turn, round_no=30, is_day=True, ours=(self.station, worker),
        )
        commands: dict = {}
        economy.worker_day(
            turn, worker, economy.Plan(), set(), set(), Memory(), commands,
            income_only=True,
        )
        self.assertEqual("sell", commands[worker.unit_id]["action"])

    def test_saves_gold_for_station_instead_of_wall_voucher(self) -> None:
        turn = replace(
            self.turn, round_no=30, is_day=True, gold=80,
            ours=(self.station, self.wall, self.worker),
        )
        self.assertEqual([], economy.shopping_list(turn))

    def test_pre_night_gunners_do_not_choose_same_step(self) -> None:
        tower_positions = (Pos(9, 25), Pos(9, 23), Pos(9, 24))
        towers = tuple(replace(tower, pos=pos) for tower, pos in
                       zip(self.turn.weapons(), tower_positions))
        first = replace(self.turn.workers()[0], pos=Pos(12, 25), backpack=())
        second = replace(self.turn.workers()[1], pos=Pos(12, 22), backpack=())
        turn = replace(
            self.turn, round_no=66, is_day=True, gold=0, zones={},
            ours=(self.station, *towers, first, second), player_tasks=(), phase_task="",
        )
        with patch.object(brain, "MEMORY", Memory()):
            commands: dict = {}
            brain._day_phase(turn, commands)
        destinations = [
            tuple(command["targetPos"][0].values())
            for command in commands.values() if command["action"] == "move"
        ]
        self.assertEqual(len(destinations), len(set(destinations)))
        self.assertTrue(destinations)

    def test_incomplete_wall_line_keeps_b_inside_while_pioneer_stages(self) -> None:
        towers = tuple(replace(tower, pos=pos) for tower, pos in zip(
            self.turn.weapons(), (Pos(9, 25), Pos(9, 23), Pos(9, 24)),
        ))
        workers = (
            replace(self.turn.workers()[0], pos=Pos(12, 25), backpack=()),
            replace(self.turn.workers()[1], pos=Pos(12, 22), backpack=()),
        )
        pioneer = replace(self.turn.pioneers()[0], pos=Pos(12, 27))
        turn = replace(
            self.turn, round_no=66, is_day=True, gold=0, zones={},
            ours=(self.station, *towers, *workers, pioneer),
            player_tasks=(), phase_task="",
        )
        with patch.object(brain, "MEMORY", Memory()):
            commands: dict = {}
            brain._day_phase(turn, commands)
            returning_roles = set(brain.MEMORY.returning_roles)
        self.assertEqual("move", commands[pioneer.unit_id]["action"])
        destinations = [
            tuple(command["targetPos"][0].values()) for command in commands.values()
            if command["action"] == "move"
        ]
        self.assertEqual(len(destinations), len(set(destinations)))
        self.assertIn(workers[-1].unit_id, returning_roles)

    def test_emergency_station_upgrade_bypasses_wall_quota(self) -> None:
        shop = Pos(6, 5)
        station = replace(self.station, health=600)
        workers = (
            replace(self.turn.workers()[0], pos=Pos(5, 5), backpack=()),
            replace(self.turn.workers()[1], pos=Pos(8, 8), backpack=()),
        )
        turn = replace(
            self.turn, round_no=30, is_day=True, gold=100,
            zones={shop: "weaponShop"}, ours=(station, *workers),
            player_tasks=(), phase_task="",
        )
        with patch.object(brain, "MEMORY", Memory()):
            commands: dict = {}
            brain._day_phase(turn, commands)
        self.assertEqual("buy", commands[workers[0].unit_id]["action"])
        self.assertEqual(STATION_UP_V1, commands[workers[0].unit_id]["name"])

    def test_no_new_task_is_started_during_night_staging(self) -> None:
        pioneer = self.turn.pioneers()[0]
        turn = replace(
            self.turn, round_no=66, is_day=True, gold=0, zones={},
            ours=(self.station, pioneer), phase_task="",
        )
        with patch.object(brain, "MEMORY", Memory()), patch.object(
            brain.tasks, "pioneer", return_value=("", ""),
        ) as task_agent:
            brain._day_phase(turn, {})
        self.assertFalse(task_agent.call_args.kwargs["allow_new_task"])

    def test_gatling_ray_hits_front_robot_not_aimed_robot(self) -> None:
        tower = replace(self.turn.weapons()[0], pos=Pos(10, 10))
        front = Robot(1, Pos(11, 10), "smallRobot", 20, False, "")
        rear = Robot(2, Pos(12, 10), "smallRobot", 20, False, "")
        remaining = {1: 20, 2: 20}
        defense._deduct_expected(tower, [rear.pos], [front, rear], remaining)
        self.assertEqual({1: 10, 2: 20}, remaining)

    def test_carried_repair_kit_is_used_without_gold_by_day_and_night(self) -> None:
        wall = replace(self.wall, pos=Pos(8, 22), health=300)
        worker = replace(self.worker, pos=Pos(8, 21), backpack=(WALL_FIXER,))
        base = replace(
            self.turn, gold=0, zones={}, ours=(self.station, wall, worker),
            player_tasks=(), phase_task="", llm_resp="", last_cmd_result="", errors=(),
        )
        with patch.object(brain, "MEMORY", Memory()):
            commands: dict = {}
            brain._day_phase(replace(base, round_no=2, is_day=True), commands)
            self.assertEqual("use", commands[worker.unit_id]["action"])
            self.assertEqual(WALL_FIXER, commands[worker.unit_id]["name"])
            commands = {}
            brain._night_phase(replace(base, round_no=71, is_day=False), commands)
            self.assertEqual("use", commands[worker.unit_id]["action"])
            self.assertEqual(WALL_FIXER, commands[worker.unit_id]["name"])

    def test_carried_station_upgrade_is_used_after_gold_is_spent(self) -> None:
        worker = replace(self.worker, pos=Pos(9, 24), backpack=(STATION_UP_V1,))
        turn = replace(
            self.turn, round_no=2, is_day=True, gold=0, zones={},
            ours=(self.station, worker), player_tasks=(), phase_task="",
        )
        with patch.object(brain, "MEMORY", Memory()):
            commands: dict = {}
            brain._day_phase(turn, commands)
        self.assertEqual("use", commands[worker.unit_id]["action"])
        self.assertEqual(STATION_UP_V1, commands[worker.unit_id]["name"])

    def test_noncritical_repair_kit_does_not_spend_upgrade_savings(self) -> None:
        turn = replace(
            self.turn, round_no=60, is_day=True, gold=10,
            ours=(self.station, self.wall, self.worker),
        )
        self.assertEqual([], economy.shopping_list(turn))

    def test_critical_wall_still_buys_emergency_repair_kit(self) -> None:
        wall = replace(self.wall, health=300)
        turn = replace(
            self.turn, round_no=60, is_day=True, gold=10,
            ours=(self.station, wall, self.worker),
        )
        self.assertEqual([(WALL_FIXER, 1)], economy.shopping_list(turn))


if __name__ == "__main__":
    unittest.main()

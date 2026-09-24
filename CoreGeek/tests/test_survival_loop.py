"""跨回合验证生产/门禁/冷却调度；此小型执行器不模拟官方机器人战斗。"""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent import brain, defense, economy, grid, protocol as P, tasks
from agent.memory import Memory


class SurvivalLoopTests(unittest.TestCase):
    def setUp(self):
        self.sample = P.Turn.load(json.loads((ROOT.parent / "docs/request.txt").read_text(encoding="utf-8")))
        self.station = self.sample.station()
        self.worker = self.sample.workers()[0]
        self.wall = self.sample.walls()[0]
        self.rocket = replace(self.sample.weapons()[0], kind=P.ROCKET, level=1,
                              cooldown=0, attack_range=10)
        self.base = replace(self.sample, round_no=1, is_day=True, gold=0, zones={},
                            ours=(self.station,), robots=(), enemy_roles=(), player_tasks=(),
                            phase_task="", llm_resp="", last_cmd_result="", errors=(),
                            world_official_news="", world_folk_legends="", last_action_results={})

    def apply_day(self, turn, commands):
        """只实现测试所用的移动、采石、建/拆墙，并验证每步的合法性。"""
        units = {unit.unit_id: unit for unit in turn.ours}
        destinations = set()
        for role_id, command in commands.items():
            role = units[role_id]
            action = command["action"]
            pos = P.Pos.load(command["targetPos"][0])
            self.assertLessEqual(P.distance(role.pos, pos), 1)
            if action == "move":
                self.assertNotIn(pos, destinations)
                self.assertNotIn(pos, turn.blocked(role))
                self.assertTrue(turn.land(pos))
                destinations.add(pos)
                units[role_id] = replace(role, pos=pos)
            elif action == "remove":
                wall = next(unit for unit in units.values() if unit.pos == pos and unit.kind == P.WALL)
                units.pop(wall.unit_id)
            elif action == "build":
                self.assertTrue(turn.is_day)
                self.assertEqual(P.WALL, command["name"])
                self.assertNotIn(pos, turn.occupied_cells())
                self.assertIn(P.STONE, role.backpack)
                backpack = list(role.backpack)
                backpack.remove(P.STONE)
                units[role_id] = replace(role, backpack=tuple(backpack))
                wall_id = max(units) + 1
                units[wall_id] = replace(self.wall, unit_id=wall_id, pos=pos, health=1000)
            elif action == "collect":
                units[role_id] = replace(role, backpack=role.backpack + (turn.zones[pos],))
            else:
                self.fail(f"unexpected command in movement fixture: {command}")
        return replace(turn, ours=tuple(units.values()))

    def test_builder_consumes_entire_batch_before_mining_again(self):
        worker = replace(self.worker, pos=P.Pos(5, 5), backpack=(P.STONE,) * 5)
        sites = tuple(grid.neighbours(worker.pos)[:5])
        turn = replace(self.base, round_no=20, ours=(self.station, worker),
                       zones={P.Pos(6, 6): P.STONE})
        memory = Memory(worker_modes={worker.unit_id: "collect"})
        for index in range(5):
            worker = turn.workers()[0]
            commands = {}
            economy.worker_day(turn, worker, economy.Plan(wall_sites=sites), set(), set(), memory, commands)
            self.assertEqual("build", commands[worker.unit_id]["action"], f"batch step {index}")
            turn = self.apply_day(turn, commands)
        self.assertEqual(0, turn.workers()[0].count(P.STONE))

    def test_two_rear_openings_remain_permanent_across_days(self):
        with patch.object(brain, "MEMORY", Memory()):
            for number, expected_walls in ((1, 10), (65, 10), (131, 10), (195, 10)):
                turn = replace(self.base, round_no=number)
                plan = brain._build_plan(turn)
                gate = brain._gate_position(turn)
                service = brain._service_gate_position(turn)
                self.assertEqual(expected_walls, len(plan.wall_sites))
                self.assertNotIn(gate, plan.wall_sites)
                self.assertNotIn(service, plan.wall_sites)
                commands = {}
                brain._day_phase(turn, commands)
                self.assertFalse(any(command["action"] == "remove"
                                     for command in commands.values()))

    def test_upper_left_layout_matches_observed_continuous_wall_strategy(self):
        station = replace(self.station, pos=P.Pos(9, 22))
        enemy = replace(self.station, unit_id=99999, pos=P.Pos(30, 9))
        turn = replace(self.base, ours=(station,), enemy_roles=(enemy,))
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(turn)
            self.assertEqual(
                (P.Pos(8, 23), P.Pos(8, 21), P.Pos(8, 20)),
                plan.tower_sites,
            )
            self.assertEqual(P.Pos(9, 23), brain.MEMORY.gunner_pos)
            self.assertEqual(P.Pos(9, 20), brain.MEMORY.repair_gunner_pos)
            self.assertEqual(P.Pos(7, 22), brain.MEMORY.gate_pos)
            self.assertEqual(P.Pos(7, 20), brain.MEMORY.service_gate_pos)
            self.assertEqual(
                {
                    P.Pos(12, 21), P.Pos(12, 22), P.Pos(12, 23),
                    P.Pos(12, 20), P.Pos(12, 24), P.Pos(12, 19),
                    P.Pos(11, 24), P.Pos(11, 19),
                    P.Pos(10, 24), P.Pos(10, 19),
                },
                set(plan.wall_sites),
            )

    def test_build_failure_is_retried_after_backoff(self):
        memory = Memory(current_round=10)
        pos = P.Pos(5, 5)
        memory.note_build_result(pos, P.WALL, False)
        self.assertFalse(memory.build_allowed(pos, P.WALL))
        memory.current_round = 30
        self.assertTrue(memory.build_allowed(pos, P.WALL))

    def test_tower_layout_preserves_access_from_gate_to_every_tower(self):
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(self.base)
            gate = brain._gate_position(self.base)
            self.assertEqual(3, len(plan.tower_sites))
            towers = tuple(replace(self.rocket, unit_id=500+i, pos=pos)
                           for i, pos in enumerate(plan.tower_sites))
            walls = tuple(replace(self.wall, unit_id=40000+i, pos=pos)
                          for i, pos in enumerate(plan.wall_sites))
            worker = replace(self.worker, pos=gate, backpack=())
            turn = replace(self.base, ours=(self.station, worker, *towers, *walls))
            for tower in towers:
                path = grid.path_to_any(turn, worker, grid.adjacent_stands(turn, worker, (tower.pos,)))
                self.assertIsNotNone(path, f"tower at {tower.pos} lost its access route")

    def test_three_rockets_use_two_rear_accessible_gunner_cells(self):
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(self.base)
            primary = brain.MEMORY.gunner_pos
            repair = brain.MEMORY.repair_gunner_pos
            self.assertIsNotNone(primary)
            self.assertIsNotNone(repair)
            self.assertNotEqual(primary, repair)
            self.assertEqual(3, len(plan.tower_sites))
            self.assertTrue(all(
                P.distance(primary, tower) <= 1 or P.distance(repair, tower) <= 1
                for tower in plan.tower_sites
            ))
            self.assertEqual(1, sum(P.distance(primary, tower) <= 1
                                    for tower in plan.tower_sites))
            self.assertGreaterEqual(sum(P.distance(repair, tower) <= 1
                                        for tower in plan.tower_sites), 2)
            self.assertNotIn(primary, plan.tower_sites)
            self.assertNotIn(repair, plan.tower_sites)

    def test_rear_openings_never_close_when_gunners_are_ready(self):
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(self.base)
            gate = brain._gate_position(self.base)
            service = brain._service_gate_position(self.base)
            ready = replace(self.base, round_no=65)
            ready_plan = brain._build_plan(ready)
            self.assertNotIn(gate, ready_plan.wall_sites)
            self.assertNotIn(service, ready_plan.wall_sites)

    def test_build_plan_never_leaves_documented_colored_rings(self):
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(self.base)
            footprint = P.station_footprint(self.station.pos)
            self.assertTrue(set(plan.tower_sites) <= set(brain._ring(footprint, 1)))
            self.assertTrue(set(plan.wall_sites) <= set(brain._ring(footprint, 2)))
            self.assertEqual(3, len(plan.tower_sites))
            self.assertEqual(10, len(plan.wall_sites))

    def test_post_wave_rocket_attacks_enemy_wall(self):
        pioneer = replace(self.sample.pioneers()[0], pos=P.Pos(5, 5), backpack=())
        rocket = replace(self.rocket, pos=P.Pos(4, 5), level=2, attack_range=15)
        enemy_wall = replace(self.wall, unit_id=900, pos=P.Pos(12, 5), health=250)
        turn = replace(
            self.base, round_no=110, is_day=False,
            ours=(self.station, pioneer, rocket), enemy_roles=(enemy_wall,), robots=(),
        )
        commands = {}
        defense.night(turn, set(), commands)
        self.assertEqual("attack", commands[rocket.unit_id]["action"])
        self.assertEqual(str(pioneer.unit_id), commands[rocket.unit_id]["controllerId"])
        self.assertEqual([enemy_wall.pos.dump(), enemy_wall.pos.dump()],
                         commands[rocket.unit_id]["targetPos"])

    def test_post_wave_rocket_prefers_enemy_station_over_wall(self):
        pioneer = replace(self.sample.pioneers()[0], pos=P.Pos(5, 5), backpack=())
        rocket = replace(self.rocket, pos=P.Pos(4, 5), level=2, attack_range=15)
        enemy_station = replace(self.station, unit_id=901, pos=P.Pos(10, 5), health=1500)
        enemy_wall = replace(self.wall, unit_id=900, pos=P.Pos(8, 5), health=1)
        turn = replace(
            self.base, round_no=110, is_day=False,
            ours=(self.station, pioneer, rocket),
            enemy_roles=(enemy_wall, enemy_station), robots=(),
        )
        commands = {}
        defense.night(turn, set(), commands)
        self.assertEqual(
            [enemy_station.pos.dump(), enemy_station.pos.dump()],
            commands[rocket.unit_id]["targetPos"],
        )

    def test_brain_does_not_waste_rocket_cooldown_before_wave_appears(self):
        pioneer = replace(self.sample.pioneers()[0], pos=P.Pos(5, 5), backpack=())
        rocket = replace(self.rocket, pos=P.Pos(4, 5), level=2, attack_range=15)
        enemy_station = replace(self.station, unit_id=901, pos=P.Pos(10, 5), health=1500)
        turn = replace(
            self.base, round_no=71, is_day=False,
            ours=(self.station, pioneer, rocket), enemy_roles=(enemy_station,), robots=(),
        )
        with patch.object(brain, "MEMORY", Memory()):
            commands = {}
            brain._night_phase(turn, commands)
        self.assertFalse(any(command["action"] == "attack" for command in commands.values()))

    def test_night_brain_reserves_workers_and_uses_pioneer_as_rotating_gunner(self):
        shared = P.Pos(5, 5)
        pioneer = replace(self.sample.pioneers()[0], pos=shared, backpack=())
        workers = tuple(replace(worker, pos=P.Pos(8 + index, 8), backpack=())
                        for index, worker in enumerate(self.sample.workers()))
        towers = tuple(replace(self.rocket, unit_id=500 + index, pos=pos, cooldown=0)
                       for index, pos in enumerate((P.Pos(4, 5), P.Pos(5, 4), P.Pos(6, 5))))
        robot = P.Robot(1, P.Pos(7, 7), P.BOSS_ROBOT, 800, False, self.base.team_type)
        turn = replace(self.base, round_no=71, is_day=False,
                       ours=(self.station, pioneer, *workers, *towers), robots=(robot,))
        with patch.object(brain, "MEMORY", Memory(gunner_pos=shared)):
            commands = {}
            brain._night_phase(turn, commands)
        attacks = [command for command in commands.values() if command["action"] == "attack"]
        self.assertEqual(1, len(attacks))
        self.assertEqual(str(pioneer.unit_id), attacks[0]["controllerId"])
        self.assertEqual("move", commands[workers[0].unit_id]["action"])
        self.assertEqual("move", commands[workers[1].unit_id]["action"])
        old_clearance = min(P.distance(workers[0].pos, item.pos) for item in turn.robots)
        new_pos = P.Pos.load(commands[workers[0].unit_id]["targetPos"][0])
        new_clearance = min(P.distance(new_pos, item.pos) for item in turn.robots)
        self.assertGreater(new_clearance, old_clearance)

    def test_unstaged_pioneer_allows_a_worker_to_fire_second_ready_rocket(self):
        shared = P.Pos(5, 5)
        first_worker = replace(self.worker, pos=P.Pos(20, 20), backpack=())
        second_worker = replace(self.sample.workers()[1], pos=shared, backpack=())
        pioneer = replace(self.sample.pioneers()[0], pos=P.Pos(7, 5), backpack=())
        towers = tuple(
            replace(self.rocket, unit_id=500 + index, pos=pos, cooldown=0)
            for index, pos in enumerate((P.Pos(4, 5), P.Pos(5, 4), P.Pos(6, 5)))
        )
        robot = P.Robot(1, P.Pos(7, 7), P.BOSS_ROBOT, 800, False, self.base.team_type)
        turn = replace(
            self.base, round_no=331, is_day=False,
            ours=(self.station, first_worker, pioneer, second_worker, *towers),
            robots=(robot,),
        )
        with patch.object(brain, "MEMORY", Memory(gunner_pos=shared)):
            commands = {}
            brain._night_phase(turn, commands)
        attacks = [command for command in commands.values() if command["action"] == "attack"]
        self.assertEqual(2, len(attacks))
        self.assertEqual(
            {str(second_worker.unit_id), str(pioneer.unit_id)},
            {command["controllerId"] for command in attacks},
        )

    def test_carried_weapon_upgrade_is_used_at_night_before_mining(self):
        rocket = replace(self.rocket, pos=P.Pos(9, 25), level=1)
        first_worker = replace(self.worker, pos=P.Pos(9, 24), backpack=())
        holder = replace(
            self.sample.workers()[1], pos=P.Pos(10, 25),
            backpack=(P.WEAPON_UP_V1,),
        )
        turn = replace(
            self.base, round_no=331, is_day=False,
            ours=(self.station, first_worker, holder, rocket), robots=(),
        )
        with patch.object(brain, "MEMORY", Memory()):
            self.assertEqual(first_worker.unit_id, brain._night_miner(turn).unit_id)
            commands = {}
            brain._night_phase(turn, commands)
        self.assertEqual("use", commands[holder.unit_id]["action"])
        self.assertEqual(P.WEAPON_UP_V1, commands[holder.unit_id]["name"])
        self.assertEqual(rocket.pos.dump(), commands[holder.unit_id]["targetPos"][0])

    def test_safe_night_miner_keeps_collecting_while_robots_exist(self):
        worker = replace(self.worker, pos=P.Pos(5, 22), backpack=())
        mine = P.Pos(6, 22)
        robot = P.Robot(1, P.Pos(20, 20), P.SMALL_ROBOT, 20, False, self.base.team_type)
        turn = replace(
            self.base, round_no=71, is_day=False, ours=(self.station, worker),
            zones={mine: P.COPPER}, robots=(robot,), vendor_prices={P.COPPER: 10},
        )
        commands = {}
        acted = economy.worker_night_safe(turn, worker, Memory(), set(), commands)
        self.assertTrue(acted)
        self.assertEqual("collect", commands[worker.unit_id]["action"])

    def test_threatened_night_miner_retreats_instead_of_collecting(self):
        worker = replace(self.worker, pos=P.Pos(10, 10), backpack=())
        mine = P.Pos(11, 10)
        robot = P.Robot(1, P.Pos(8, 8), P.SMALL_ROBOT, 20, False, self.base.team_type)
        turn = replace(
            self.base, round_no=71, is_day=False, ours=(self.station, worker),
            zones={mine: P.COPPER}, robots=(robot,), vendor_prices={P.COPPER: 10},
        )
        commands = {}
        economy.worker_night_safe(turn, worker, Memory(), set(), commands)
        self.assertEqual("move", commands[worker.unit_id]["action"])
        step = P.Pos.load(commands[worker.unit_id]["targetPos"][0])
        self.assertGreater(P.distance(step, robot.pos), P.distance(worker.pos, robot.pos))

    def test_night_miner_keeps_retreat_mode_instead_of_reversing_to_mine(self):
        worker = replace(self.worker, pos=P.Pos(10, 10), backpack=())
        mine = P.Pos(11, 10)
        robot = P.Robot(1, P.Pos(8, 8), P.SMALL_ROBOT, 20, False, self.base.team_type)
        memory = Memory(current_round=71)
        turn = replace(
            self.base, round_no=71, is_day=False, ours=(self.station, worker),
            zones={mine: P.COPPER}, robots=(robot,), vendor_prices={P.COPPER: 10},
        )
        commands = {}
        self.assertTrue(economy.worker_night_safe(turn, worker, memory, set(), commands))
        first = P.Pos.load(commands[worker.unit_id]["targetPos"][0])
        worker = replace(worker, pos=first)
        memory.current_round = 72
        memory.note_positions((worker,))
        turn = replace(turn, round_no=72, ours=(self.station, worker))
        commands = {}
        self.assertTrue(economy.worker_night_safe(turn, worker, memory, set(), commands))
        self.assertEqual("move", commands[worker.unit_id]["action"])
        second = P.Pos.load(commands[worker.unit_id]["targetPos"][0])
        self.assertNotEqual(P.Pos(10, 10), second)

    def test_opponent_wave_does_not_block_our_post_wave_state(self):
        other_team = "defender" if self.base.team_type == "challenger" else "challenger"
        our_robot = P.Robot(1, P.Pos(20, 20), P.SMALL_ROBOT, 40, False,
                            self.base.team_type)
        other_robot = replace(our_robot, robot_id=2, target_team=other_team)
        with patch.object(brain, "MEMORY", Memory()):
            self.assertFalse(brain._post_wave_safe(
                replace(self.base, round_no=89, is_day=False, robots=(our_robot,))))
            self.assertFalse(brain._post_wave_safe(
                replace(self.base, round_no=90, is_day=False, robots=(other_robot,))))
            self.assertTrue(brain._post_wave_safe(
                replace(self.base, round_no=91, is_day=False, robots=(other_robot,))))

    def test_active_task_recovery_never_walks_to_shop(self):
        pioneer = replace(self.sample.pioneers()[0], health=40, backpack=())
        turn = replace(self.base, ours=(self.station, pioneer))
        commands = {}
        self.assertFalse(brain._recover_pioneer(
            turn, pioneer, set(), commands, allow_travel=False,
        ))
        self.assertEqual({}, commands)

        pioneer = replace(pioneer, backpack=(P.MEDICINE,))
        commands = {}
        self.assertTrue(brain._recover_pioneer(
            turn, pioneer, set(), commands, allow_travel=False,
        ))
        self.assertEqual("use", commands[pioneer.unit_id]["action"])
        self.assertNotIn("targetPos", commands[pioneer.unit_id])

    def test_cancelled_decision_discards_candidate_memory(self):
        live = Memory(current_round=12)

        def fake_decide(_payload):
            brain.MEMORY.current_round = 99
            return {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}

        with patch.object(brain, "MEMORY", live), patch.object(brain, "decide", fake_decide):
            self.assertFalse(brain.decide_transactional({}, lambda _result: False))
            self.assertIs(brain.MEMORY, live)
            self.assertEqual(12, brain.MEMORY.current_round)

    def test_match_identity_change_resets_cross_match_memory(self):
        stale = Memory(
            current_round=500,
            match_team=self.base.team_type,
            match_station_pos=self.station.pos,
            task_state="exploring",
        )
        moved_station = replace(
            self.station, pos=P.Pos(self.station.pos.x + 1, self.station.pos.y),
        )
        new_turn = replace(self.base, round_no=501, ours=(moved_station,))
        with patch.object(brain, "MEMORY", stale):
            self.assertTrue(brain._reset_match_if_needed(new_turn))
            self.assertEqual("idle", brain.MEMORY.task_state)
            self.assertEqual(moved_station.pos, brain.MEMORY.match_station_pos)

    def test_night_sell_without_target_position_does_not_crash(self):
        worker = replace(
            self.worker, pos=P.Pos(5, 5), backpack=(P.COPPER,) * 8, capacity=10,
        )
        turn = replace(
            self.base, round_no=71, is_day=False, ours=(self.station, worker),
            zones={P.Pos(6, 5): "vendor"}, vendor_prices={P.COPPER: 10}, robots=(),
        )
        commands = {}
        self.assertTrue(economy.worker_night_safe(turn, worker, Memory(), set(), commands))
        self.assertEqual("sell", commands[worker.unit_id]["action"])
        self.assertNotIn("targetPos", commands[worker.unit_id])

    def test_post_wave_a_repairs_while_b_keeps_mining(self):
        first = replace(self.worker, pos=P.Pos(8, 8), backpack=(), capacity=10)
        second = replace(
            self.sample.workers()[1], pos=P.Pos(5, 5),
            backpack=(P.WALL_FIXER,), capacity=10,
        )
        wall = replace(self.wall, pos=P.Pos(5, 6), health=200)
        mine = P.Pos(9, 8)
        turn = replace(
            self.base, round_no=100, is_day=False,
            ours=(self.station, first, second, wall), robots=(),
            zones={mine: P.COPPER}, vendor_prices={P.COPPER: 10},
        )
        memory = Memory(
            wave_seen_days={1}, wave_clear_streak={1: 1},
            wave_clear_observed_round={1: 99},
        )
        with patch.object(brain, "MEMORY", memory):
            commands = {}
            brain._night_phase(turn, commands)
        self.assertEqual("use", commands[second.unit_id]["action"])
        self.assertEqual(P.WALL_FIXER, commands[second.unit_id]["name"])
        self.assertEqual("collect", commands[first.unit_id]["action"])

    def test_failed_move_is_temporarily_blacklisted(self):
        target = P.Pos(self.worker.pos.x + 1, self.worker.pos.y)
        memory = Memory(
            current_round=20,
            last_commands={str(self.worker.unit_id): P.move_command(target)},
        )
        turn = replace(
            self.base, round_no=21, ours=(self.station, self.worker),
            last_action_results={self.worker.unit_id: False},
        )
        with patch.object(brain, "MEMORY", memory):
            brain._learn_from_feedback(turn)
        self.assertIn(target, memory.failed_move_cells(self.worker.unit_id))

    def test_gate_is_behind_base_and_front_walls_are_built_first(self):
        enemy_station = replace(
            self.station, unit_id=99999,
            pos=P.Pos(self.station.pos.x + 15, self.station.pos.y),
        )
        turn = replace(self.base, enemy_roles=(enemy_station,))
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(turn)
            gate = brain._gate_position(turn)
            front = brain._front_wall_sites(turn)
            self.assertNotIn(gate, front)
            ring = brain._ring(P.station_footprint(self.station.pos), 2)
            self.assertEqual(
                min(brain._enemy_projection(turn, pos) for pos in ring),
                min(brain._enemy_projection(turn, gate),
                    brain._enemy_projection(turn, brain._service_gate_position(turn))),
            )
            self.assertNotEqual(gate, brain._service_gate_position(turn))
            self.assertLess(
                brain._enemy_projection(turn, gate),
                min(brain._enemy_projection(turn, pos) for pos in front),
            )
            self.assertEqual(
                max(brain._enemy_projection(turn, pos) for pos in plan.wall_sites),
                brain._enemy_projection(turn, plan.wall_sites[0]),
            )

    def test_three_rocket_battery_is_rear_mounted_and_covers_front_from_every_corner(self):
        corners = (
            (P.Pos(3, 4), P.Pos(35, 27)),
            (P.Pos(35, 4), P.Pos(3, 27)),
            (P.Pos(3, 27), P.Pos(35, 4)),
            (P.Pos(35, 27), P.Pos(3, 4)),
        )
        for station_pos, enemy_pos in corners:
            with self.subTest(station=station_pos), patch.object(brain, "MEMORY", Memory()):
                station = replace(self.station, pos=station_pos)
                enemy = replace(self.station, unit_id=99999, pos=enemy_pos)
                turn = replace(
                    self.base, ours=(station,), enemy_roles=(enemy,), zones={},
                )
                plan = brain._build_plan(turn)
                projections = [brain._enemy_projection(turn, pos) for pos in plan.tower_sites]
                self.assertEqual(3, len(plan.tower_sites))
                self.assertLess(max(projections), 0)
                self.assertEqual(1, sum(
                    P.distance(brain.MEMORY.gunner_pos, pos) <= 1
                    for pos in plan.tower_sites
                ))
                self.assertGreaterEqual(sum(
                    P.distance(brain.MEMORY.repair_gunner_pos, pos) <= 1
                    for pos in plan.tower_sites
                ), 2)
                self.assertTrue(all(
                    all(P.distance(tower, wall) <= 10 for tower in plan.tower_sites)
                    for wall in brain._front_wall_sites(turn)
                ))
                self.assertNotIn(brain.MEMORY.gate_pos, brain._front_wall_sites(turn))
                self.assertNotIn(brain.MEMORY.service_gate_pos, brain._front_wall_sites(turn))

    def test_first_worker_mines_while_second_worker_builds(self):
        with patch.object(brain, "MEMORY", Memory()):
            plan = brain._build_plan(self.base)
            walls = tuple(
                replace(self.wall, unit_id=41000 + index, pos=pos, health=1000)
                for index, pos in enumerate(plan.wall_sites[:4])
            )
            workers = tuple(
                replace(worker, pos=P.Pos(5 + index, 5), backpack=())
                for index, worker in enumerate(self.sample.workers())
            )
            turn = replace(
                self.base, round_no=20, ours=(self.station, *workers, *walls), zones={},
            )
            with patch.object(economy, "worker_day") as worker_day:
                brain._day_phase(turn, {})
        self.assertEqual(2, worker_day.call_count)
        by_worker = {call.args[1].unit_id: call.kwargs["income_only"]
                     for call in worker_day.call_args_list}
        self.assertTrue(by_worker[min(by_worker)])
        self.assertFalse(by_worker[max(by_worker)])

    def test_one_role_rotates_three_rockets_by_reported_cooldown(self):
        worker = replace(self.worker, pos=P.Pos(5, 5), backpack=())
        robots = (P.Robot(1, P.Pos(7, 7), P.BOSS_ROBOT, 800, False, self.base.team_type),)
        towers = tuple(replace(self.rocket, unit_id=500+i, pos=pos)
                       for i, pos in enumerate((P.Pos(4, 5), P.Pos(5, 4), P.Pos(6, 5))))
        fired = []
        for active in range(3):
            current = tuple(replace(t, cooldown=0 if i == active else 2) for i, t in enumerate(towers))
            turn = replace(self.base, round_no=71+active, is_day=False,
                           ours=(self.station, worker, *current), robots=robots)
            commands = {}
            defense.night(turn, set(), commands)
            attacks = [(key, value) for key, value in commands.items() if value["action"] == "attack"]
            self.assertEqual(1, len(attacks))
            self.assertEqual(str(worker.unit_id), attacks[0][1]["controllerId"])
            self.assertNotIn(worker.unit_id, commands)
            fired.append(attacks[0][0])
        self.assertEqual([500, 501, 502], fired)

    def test_cooldown_allows_adjacent_repair(self):
        worker = replace(self.worker, pos=P.Pos(5, 5), backpack=(P.WALL_FIXER,))
        wall = replace(self.wall, pos=P.Pos(5, 6), health=650)
        tower = replace(self.rocket, pos=P.Pos(6, 5), cooldown=2)
        turn = replace(self.base, round_no=71, is_day=False, ours=(self.station, worker, wall, tower))
        commands = {}
        defense.night(turn, set(), commands)
        self.assertEqual(P.WALL_FIXER, commands[worker.unit_id]["name"])

    def test_two_repairers_do_not_spend_two_kits_on_same_wall(self):
        first = replace(self.worker, pos=P.Pos(5, 5), backpack=(P.WALL_FIXER,))
        second = replace(self.sample.workers()[1], pos=P.Pos(6, 5), backpack=(P.WALL_FIXER,))
        wall = replace(self.wall, pos=P.Pos(5, 6), health=250)
        turn = replace(self.base, round_no=71, is_day=False, ours=(self.station, first, second, wall))
        commands, claimed = {}, set()
        for role in (first, second):
            economy.move_or_repair_wall(turn, role, claimed, commands, urgent_only=True)
        self.assertEqual(1, sum(c["action"] == "use" for c in commands.values()))

    def test_repair_does_not_ignore_reachable_wall_for_remote_worse_wall(self):
        worker = replace(self.worker, pos=P.Pos(5, 5), backpack=(P.WALL_FIXER,))
        nearby = replace(self.wall, pos=P.Pos(5, 6), health=250)
        remote = replace(self.wall, unit_id=999, pos=P.Pos(25, 25), health=1)
        turn = replace(self.base, is_day=False, ours=(self.station, worker, nearby, remote))
        commands = {}
        self.assertTrue(economy.move_or_repair_wall(turn, worker, set(), commands, urgent_only=True))
        self.assertEqual(nearby.pos.dump(), commands[worker.unit_id]["targetPos"][0])

    def test_upgrade_priority_changes_with_station_health(self):
        rockets = tuple(replace(self.rocket, unit_id=500 + index,
                                pos=P.Pos(5 + index, 5)) for index in range(3))
        turn = replace(self.base, round_no=30, gold=100, ours=(self.station, *rockets))
        self.assertEqual(P.WEAPON_UP_V1, economy.shopping_list(turn)[0][0])
        turn = replace(turn, ours=(replace(self.station, health=640), *rockets))
        self.assertEqual(P.STATION_UP_V1, economy.shopping_list(turn)[0][0])

    def test_three_two_one_battery_inserts_station_insurance_when_damaged(self):
        levels = (3, 2, 1)
        rockets = tuple(
            replace(
                self.rocket, unit_id=500 + index, pos=P.Pos(5 + index, 5),
                level=level, health=economy._wall_max_health(level),
            )
            for index, level in enumerate(levels)
        )
        turn = replace(
            self.base, round_no=140, gold=150,
            ours=(replace(self.station, health=1250), *rockets),
        )
        self.assertEqual(P.STATION_UP_V1, economy.shopping_list(turn)[0][0])

    def test_three_level_one_rockets_make_weapon_upgrade_first(self):
        rockets = tuple(
            replace(self.rocket, unit_id=500 + index, pos=P.Pos(5 + index, 5))
            for index in range(3)
        )
        turn = replace(self.base, round_no=30, gold=100, ours=(self.station, *rockets))
        self.assertEqual(P.WEAPON_UP_V1, economy.shopping_list(turn)[0][0])

    def test_carried_station_upgrade_can_save_base_at_night(self):
        station = replace(self.station, health=640)
        worker = replace(self.worker, pos=P.Pos(9, 24), backpack=(P.STATION_UP_V1,))
        turn = replace(self.base, round_no=201, is_day=False, ours=(station, worker))
        with patch.object(brain, "MEMORY", Memory()):
            commands = {}
            brain._night_phase(turn, commands)
        self.assertEqual(P.STATION_UP_V1, commands[worker.unit_id]["name"])

    def test_imminent_threat_beats_distant_same_size_cluster(self):
        near = P.Robot(1, P.Pos(12, 24), P.SMALL_ROBOT, 40, False, self.base.team_type)
        far = replace(near, robot_id=2, pos=P.Pos(30, 30))
        hp = {1: 40, 2: 40}
        self.assertGreater(defense._rocket_value(self.base, near.pos, [near], hp),
                           defense._rocket_value(self.base, far.pos, [far], hp))

    def test_reserved_step_is_avoided_during_path_search(self):
        worker = replace(self.worker, pos=P.Pos(5, 5), backpack=())
        turn = replace(self.base, ours=(self.station, worker))
        first = grid.next_step_adjacent(turn, worker, P.Pos(9, 5))
        alternate = grid.next_step_adjacent(turn, worker, P.Pos(9, 5), reserved={first})
        self.assertIsNotNone(alternate)
        self.assertNotEqual(first, alternate)

    def test_move_conflicts_allow_train_but_reject_direct_swap(self):
        first = replace(self.worker, pos=P.Pos(5, 5), backpack=())
        second = replace(self.sample.workers()[1], pos=P.Pos(6, 5), backpack=())
        turn = replace(self.base, ours=(self.station, first, second))
        with patch.object(brain, "MEMORY", Memory()):
            train = {
                first.unit_id: P.move_command(second.pos),
                second.unit_id: P.move_command(P.Pos(7, 5)),
            }
            brain._resolve_command_conflicts(turn, train)
            self.assertEqual({first.unit_id, second.unit_id}, set(train))

            swap = {
                first.unit_id: P.move_command(second.pos),
                second.unit_id: P.move_command(first.pos),
            }
            brain._resolve_command_conflicts(turn, swap)
            self.assertEqual({}, swap)

    def test_partial_answer_submitted_then_completed_without_reusing_wrong_answer(self):
        role = replace(self.sample.pioneers()[0], pos=P.Pos(13, 14))
        memory = Memory(task_state="exploring", task_point=P.Pos(14, 14), task_started_round=1,
                        task_desc="Return count and token", task_steps_tried=3,
                        task_verified_tool_output=True, pending_prompt_kind="task", pending_prompt_round=4)
        turn = replace(self.base, round_no=5, ours=(self.station, role), phase_task=memory.task_desc,
                       llm_resp='FINAL: {"count":6}')
        commands = {}
        tasks.pioneer(turn, role, memory, set(), commands, allow_new_task=False)
        self.assertEqual('{"count":6}', commands[role.unit_id]["taskAnswer"])
        turn = replace(turn, round_no=6, llm_resp="", errors=(P.GameError(2, "missing token"),))
        prompt, _ = tasks.pioneer(turn, role, memory, set(), {}, allow_new_task=False)
        self.assertIn("部分", prompt)
        self.assertIn("missing token", prompt)
        turn = replace(turn, round_no=7, errors=(), llm_resp='FINAL: {"count":6,"token":"verified"}')
        commands = {}
        tasks.pioneer(turn, role, memory, set(), commands, allow_new_task=False)
        self.assertEqual("submitAnswer", commands[role.unit_id]["action"])
        self.assertEqual([], memory.sops)

    def test_departure_deadline_ends_unfinished_task_for_return(self):
        role = replace(self.sample.pioneers()[0], pos=P.Pos(13, 14))
        memory = Memory(task_state="exploring", task_point=P.Pos(14, 14), task_started_round=1,
                        task_desc="query", task_steps_tried=3)
        turn = replace(self.base, round_no=60, ours=(self.station, role), phase_task="query")
        commands = {}
        tasks.pioneer(turn, role, memory, set(), commands, allow_new_task=False, departure_round=60)
        self.assertEqual("idle", memory.task_state)
        self.assertEqual({}, commands)

    def test_task_selection_accounts_for_travel_and_return_time(self):
        role = replace(self.sample.pioneers()[0], pos=P.Pos(9, 24))
        nearby = replace(self.sample.player_tasks[0], position=P.Pos(7, 24),
                         timeout_rounds=20, cold_down=0, is_valid=True, score_reward=10)
        remote = replace(nearby, position=P.Pos(30, 20), score_reward=1000)
        turn = replace(self.base, round_no=40, ours=(self.station, role), player_tasks=(nearby, remote))
        self.assertEqual(nearby.position, tasks._pick_task_point(turn, role, Memory()))

    def test_post_wave_night_window_can_start_a_finishable_task(self):
        role = replace(self.sample.pioneers()[0], pos=P.Pos(9, 24))
        nearby = replace(
            self.sample.player_tasks[0], position=P.Pos(7, 24),
            timeout_rounds=20, cold_down=0, is_valid=True, score_reward=100,
        )
        turn = replace(
            self.base, round_no=112, is_day=False,
            ours=(self.station, role), player_tasks=(nearby,),
        )
        self.assertEqual(
            nearby.position,
            tasks._pick_task_point(turn, role, Memory(), available_rounds=80),
        )

    def test_task_approach_keeps_locked_destination(self):
        role = replace(self.sample.pioneers()[0], pos=P.Pos(9, 24))
        first = replace(self.sample.player_tasks[0], position=P.Pos(7, 24),
                        timeout_rounds=20, cold_down=0, is_valid=True, score_reward=10)
        second = replace(first, position=P.Pos(14, 24), score_reward=1000)
        turn = replace(self.base, round_no=10, ours=(self.station, role),
                       player_tasks=(first, second))
        memory = Memory(task_approach_point=first.position)
        self.assertEqual(first.position, tasks._pick_task_point(turn, role, memory))

    def test_too_short_task_is_skipped_instead_of_timing_out(self):
        role = replace(self.sample.pioneers()[0], pos=P.Pos(9, 24))
        short = replace(self.sample.player_tasks[0], position=P.Pos(7, 24),
                        timeout_rounds=8, cold_down=0, is_valid=True)
        turn = replace(self.base, round_no=10, ours=(self.station, role), player_tasks=(short,))
        self.assertIsNone(tasks._pick_task_point(turn, role, Memory()))


if __name__ == "__main__":
    unittest.main()

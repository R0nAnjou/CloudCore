#!/usr/bin/env python3
"""关键规则回归测试：覆盖曾导致不行动、非法攻击和夜战崩溃的问题。"""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import defense, tasks  # noqa: E402
from agent.grid import next_step_adjacent  # noqa: E402
from agent.memory import Memory  # noqa: E402
from agent.protocol import GameError, Pos, Robot, Turn  # noqa: E402


class RegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        payload = json.loads((ROOT.parent / "docs" / "request.txt").read_text(encoding="utf-8"))
        self.payload = payload
        self.turn = Turn.load(payload)

    def test_interaction_path_targets_adjacent_land(self) -> None:
        worker = self.turn.workers()[0]
        pioneer = self.turn.pioneers()[0]
        vendor_step = next_step_adjacent(self.turn, worker, self.turn.vendor_pos())
        task_step = next_step_adjacent(self.turn, pioneer, self.turn.player_tasks[0].position)
        self.assertIsNotNone(vendor_step)
        self.assertIsNotNone(task_step)
        self.assertEqual(1, max(abs(vendor_step.x - worker.pos.x), abs(vendor_step.y - worker.pos.y)))

    def test_enemy_buildings_are_blocked(self) -> None:
        worker = self.turn.workers()[0]
        blocked = self.turn.blocked(worker)
        enemy_station = next(unit for unit in self.turn.enemy_roles if unit.kind == "station")
        self.assertTrue(set(self.turn.foot_print(enemy_station)) <= blocked)

    def test_night_attack_does_not_mutate_frozen_robots(self) -> None:
        payload = self.payload
        payload["roundNo"] = 71
        payload["teamOur"]["roles"][4]["pos"] = {"x": 8, "y": 25}
        payload["robot"]["roles"][0]["pos"] = {"x": 8, "y": 24}
        turn = Turn.load(payload)
        original_hp = tuple(robot.health for robot in turn.robots)
        commands = {}
        defense.night(turn, set(), commands)
        self.assertTrue(any(cmd["action"] == "attack" for cmd in commands.values()))
        self.assertEqual(original_hp, tuple(robot.health for robot in turn.robots))

    def test_upgraded_rocket_emits_exact_target_count(self) -> None:
        tower = next(unit for unit in self.turn.weapons() if unit.kind == "rocket")
        tower = replace(tower, level=3, attack_range=10**9)
        targets = defense._choose_targets(self.turn, tower, list(self.turn.robots))
        self.assertEqual(3, len(targets))

    def test_gatling_targets_are_pairwise_inside_cone(self) -> None:
        tower = next(unit for unit in self.turn.weapons() if unit.kind == "gatling")
        tower = replace(tower, pos=Pos(20, 16), level=3, attack_range=20)
        robots = [
            Robot(1, Pos(25, 16), "smallRobot", 40, False, self.turn.team_type),
            Robot(2, Pos(21, 11), "smallRobot", 40, False, self.turn.team_type),
            Robot(3, Pos(21, 21), "smallRobot", 40, False, self.turn.team_type),
        ]
        targets = defense._choose_targets(self.turn, tower, robots)
        self.assertEqual(3, len(targets))
        for index, first in enumerate(targets):
            for second in targets[index + 1:]:
                self.assertTrue(defense._within_90deg(tower.pos, first, second))

    def test_task_state_resets_after_server_ends_task(self) -> None:
        memory = Memory(task_state="accepting", task_started_round=84, task_point=Pos(14, 14))
        accepted_turn = replace(self.turn, round_no=85, phase_task="读取接口并回答")
        role = accepted_turn.pioneers()[0]
        tasks._sync_task_state(accepted_turn, role, memory)
        self.assertEqual("accepted", memory.task_state)
        ended_turn = replace(accepted_turn, round_no=87, phase_task="")
        tasks._sync_task_state(ended_turn, role, memory)
        self.assertEqual("idle", memory.task_state)

    def test_task_pipeline_accumulates_output_and_submits_answer(self) -> None:
        task_pos = Pos(14, 14)
        role = replace(self.turn.pioneers()[0], pos=Pos(13, 14))
        turn = replace(
            self.turn,
            round_no=2,
            ours=tuple(role if unit.kind == "pioneer" else unit for unit in self.turn.ours),
            phase_task="读取本地接口文档并返回城市天气",
            errors=(),
        )
        memory = Memory(task_state="accepting", task_started_round=1, task_point=task_pos)
        commands = {}
        prompt, execute = tasks.pioneer(turn, role, memory, set(), commands)
        self.assertEqual("exploring", memory.task_state)
        self.assertTrue(execute)

        turn = replace(turn, round_no=3, last_cmd_result="[exitCode:0]\napi.md")
        prompt, execute = tasks.pioneer(turn, role, memory, set(), {})
        self.assertTrue(execute)
        turn = replace(turn, round_no=4, last_cmd_result="[exitCode:0]\n接口内容")
        prompt, execute = tasks.pioneer(turn, role, memory, set(), {})
        self.assertTrue(prompt)
        self.assertIn("api.md", prompt)
        self.assertIn("接口内容", prompt)

        turn = replace(turn, round_no=5, llm_resp="晴，25°C", last_cmd_result="")
        commands = {}
        tasks.pioneer(turn, role, memory, set(), commands)
        self.assertEqual("submitAnswer", commands[role.unit_id]["action"])
        self.assertEqual("晴，25°C", commands[role.unit_id]["taskAnswer"])
        self.assertEqual([], memory.sops)

    def test_task_filename_with_chinese_punctuation_is_read(self) -> None:
        desc = "请阅读task_1_alpha.md，获取任务信息"
        self.assertEqual("task_1_alpha.md", tasks._extract_local_target(desc))
        memory = Memory(task_steps_tried=1)
        self.assertIn("-name 'task_1_alpha.md'", tasks._next_explore_command(memory, desc))

    def test_rejected_answer_triggers_more_exploration_not_sop(self) -> None:
        desc = "请阅读task_1_alpha.md，获取任务信息"
        role = replace(self.turn.pioneers()[0], pos=Pos(13, 14))
        turn = replace(
            self.turn, round_no=6, phase_task=desc,
            errors=(GameError(2, "答案缺少字段"),),
        )
        memory = Memory(
            task_state="answering", task_started_round=2, task_point=Pos(14, 14),
            task_desc=desc, task_steps_tried=2, task_answer="错误答案",
            task_answered_round=5,
        )
        commands = {}
        prompt, execute = tasks.pioneer(turn, role, memory, set(), commands)
        self.assertEqual("", prompt)
        self.assertIn("241,800p", execute)
        self.assertEqual([], memory.sops)
        self.assertEqual(["错误答案"], memory.task_rejected_answers)
        self.assertNotIn(role.unit_id, commands)
        self.assertIn("答案缺少字段", memory.task_transcript[-1])

    def test_verified_answer_is_saved_only_after_task_ends(self) -> None:
        desc = "请阅读task_1_alpha.md，获取任务信息"
        role = replace(self.turn.pioneers()[0], pos=Pos(13, 14))
        memory = Memory(
            task_state="answering", task_started_round=2, task_point=Pos(14, 14),
            task_desc=desc, task_answer="正确答案", task_answered_round=5,
        )
        pending = replace(self.turn, round_no=5, phase_task=desc, errors=())
        tasks._sync_task_state(pending, role, memory)
        self.assertEqual([], memory.sops)
        completed = replace(pending, round_no=6, phase_task="")
        tasks._sync_task_state(completed, role, memory)
        self.assertEqual("idle", memory.task_state)
        self.assertEqual("正确答案", memory.sops[0].answer_hint)

    def test_task_timeout_never_saves_unverified_answer(self) -> None:
        role = replace(self.turn.pioneers()[0], pos=Pos(13, 14))
        memory = Memory(
            task_state="answering", task_started_round=2, task_point=Pos(14, 14),
            task_desc="任务", task_answer="待验证答案", task_answered_round=5,
        )
        timed_out = replace(
            self.turn, round_no=6, phase_task="", errors=(GameError(1, "任务超时"),),
        )
        tasks._sync_task_state(timed_out, role, memory)
        self.assertEqual("idle", memory.task_state)
        self.assertEqual([], memory.sops)

    def test_repeated_llm_answer_is_not_resubmitted(self) -> None:
        desc = "请阅读task_1_alpha.md，获取任务信息"
        role = replace(self.turn.pioneers()[0], pos=Pos(13, 14))
        memory = Memory(
            task_state="exploring", task_started_round=2, task_point=Pos(14, 14),
            task_desc=desc, task_steps_tried=4,
            task_rejected_answers=["错误答案"],
            pending_prompt_round=5, pending_prompt_kind="task",
        )
        turn = replace(
            self.turn, round_no=6, phase_task=desc, llm_resp="错误答案", errors=(),
        )
        commands = {}
        prompt, execute = tasks.pioneer(turn, role, memory, set(), commands)
        self.assertNotIn(role.unit_id, commands)
        self.assertEqual("", execute)
        self.assertIn("错误答案", prompt)

    def test_similar_sop_still_reads_current_task_material(self) -> None:
        desc = "请阅读task_2_beta.md，获取任务信息"
        role = replace(self.turn.pioneers()[0], pos=Pos(13, 14))
        memory = Memory(
            task_state="accepted", task_started_round=2, task_point=Pos(14, 14),
            task_desc=desc,
        )
        memory.add_sop("请阅读task_1_alpha.md，获取任务信息", [], "上次答案")
        turn = replace(self.turn, round_no=3, phase_task=desc, errors=())
        commands = {}
        prompt, execute = tasks.pioneer(turn, role, memory, set(), commands)
        self.assertEqual("", prompt)
        self.assertTrue(execute)
        self.assertNotIn(role.unit_id, commands)

    def test_task_supplies_are_derived_from_shop(self) -> None:
        supplies = tasks.task_supply_names(self.turn)
        self.assertIn("AcientTablet", supplies)
        self.assertNotIn("Medicine", supplies)

    def test_treasure_time_parser_distinguishes_day_from_round(self) -> None:
        day_two_night = replace(self.turn, round_no=201, is_day=False)
        self.assertTrue(tasks._treasure_time_matches(day_two_night, "DAY2 夜晚"))
        self.assertFalse(tasks._treasure_time_matches(day_two_night, "DAY3 夜晚"))
        self.assertFalse(tasks._treasure_time_matches(day_two_night, "第300-320回合"))
        self.assertTrue(tasks._treasure_time_matches(day_two_night, "第190-220回合 夜晚"))


if __name__ == "__main__":
    unittest.main()

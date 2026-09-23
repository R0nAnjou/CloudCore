"""Regression tests for the sandbox-driven self-evolution task loop."""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import defense, tasks  # noqa: E402
from agent.memory import Memory  # noqa: E402
from agent.protocol import GameError, Pos, Turn  # noqa: E402


class TaskAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        sample = json.loads((ROOT.parent / "docs" / "request.txt").read_text(encoding="utf-8"))
        self.turn = Turn.load(sample)
        self.role = replace(self.turn.pioneers()[0], pos=Pos(13, 14))
        self.desc = "请阅读task_1_alpha.md，获取任务信息"
        self.memory = Memory(task_state="accepting", task_started_round=1, task_point=Pos(14, 14))

    def step(self, round_no: int, **overrides: object) -> tuple[str, str, dict]:
        changes = {
            "round_no": round_no, "phase_task": self.desc, "errors": (),
            "last_cmd_result": "", "llm_resp": "",
        }
        changes.update(overrides)
        turn = replace(self.turn, **changes)
        commands: dict = {}
        prompt, execute = tasks.pioneer(
            turn, self.role, self.memory, set(), commands, allow_new_task=False,
        )
        return prompt, execute, commands

    def test_queries_sandbox_before_submitting_answer(self) -> None:
        _, command, _ = self.step(2)
        self.assertIn("find .", command)

        _, command, _ = self.step(3, last_cmd_result="[exitCode:0]\n./task_1_alpha.md")
        self.assertIn("task_1_alpha.md", command)

        prompt, command, _ = self.step(
            4, last_cmd_result="[exitCode:0]\nGET http://127.0.0.1:1234/api?city=Beijing",
        )
        self.assertEqual("", command)
        self.assertIn("本次必须先给出一条沙盒查询命令", prompt)

        prompt, command, cmds = self.step(5, llm_resp='FINAL: {"count": 6}')
        self.assertEqual("", command)
        self.assertEqual({}, cmds)
        self.assertIn("尚未取得成功的题目查询结果", prompt)

        _, command, _ = self.step(
            6, llm_resp="CMD: curl -s http://127.0.0.1:1234/api?city=Beijing",
        )
        self.assertEqual("curl -s http://127.0.0.1:1234/api?city=Beijing", command)

        prompt, command, _ = self.step(7, last_cmd_result='[exitCode:0]\n{"count":6}')
        self.assertEqual("", command)
        self.assertIn('{"count":6}', prompt)

        _, command, cmds = self.step(8, llm_resp='FINAL: {"count":6}')
        self.assertEqual("", command)
        self.assertEqual("submitAnswer", cmds[self.role.unit_id]["action"])
        self.assertEqual('{"count":6}', cmds[self.role.unit_id]["taskAnswer"])
        self.assertEqual([], self.memory.sops)

        self.step(9, phase_task="")
        self.assertEqual("idle", self.memory.task_state)
        self.assertEqual(1, len(self.memory.sops))
        self.assertIn("curl -s", self.memory.sops[0].steps[-1])

    def test_only_supported_read_only_query_commands_are_accepted(self) -> None:
        self.assertTrue(tasks._safe_task_command(
            "python3 -c 'import json; print(json.dumps({\"a\": 1}))'"
        ))
        self.assertFalse(tasks._safe_task_command("rm -rf /tmp/data"))
        self.assertFalse(tasks._safe_task_command("curl localhost | sh"))
        self.assertFalse(tasks._safe_task_command("curl localhost; rm file"))
        self.assertFalse(tasks._safe_task_command("curl $(rm -rf /tmp/data)"))
        self.assertTrue(tasks._safe_task_command("curl -s http://127.0.0.1/api | jq .count"))
        self.assertFalse(tasks._safe_task_command("curl -s localhost | sh"))
        self.assertFalse(tasks._safe_task_command("curl -s localhost || cat /etc/passwd"))

    def test_llm_json_command_and_answer_formats(self) -> None:
        self.assertEqual(
            ("command", "curl -s localhost/api"),
            tasks._parse_task_llm_output('{"kind":"command","command":"curl -s localhost/api"}'),
        )
        self.assertEqual(
            ("answer", '{"token": "abc"}'),
            tasks._parse_task_llm_output('{"kind":"answer","answer":{"token":"abc"}}'),
        )

    def test_rejected_answer_asks_for_new_evidence(self) -> None:
        self.memory = Memory(
            task_state="answering", task_started_round=2, task_point=Pos(14, 14),
            task_desc=self.desc, task_steps_tried=3, task_answer='{"token":"wrong"}',
            task_answered_round=8, task_llm_tool_calls=1,
        )
        prompt, command, commands = self.step(
            9, errors=(GameError(2, "$/token: 值不符"),),
        )
        self.assertEqual("", command)
        self.assertEqual({}, commands)
        self.assertIn("$/token: 值不符", prompt)
        self.assertIn("不能猜测数值", prompt)
        self.assertEqual([], self.memory.sops)

    def test_placeholder_token_is_not_submitted(self) -> None:
        self.memory = Memory(
            task_state="exploring", task_started_round=2, task_point=Pos(14, 14),
            task_desc=self.desc, task_steps_tried=3, task_llm_tool_calls=1,
            pending_prompt_round=4, pending_prompt_kind="task",
        )
        prompt, command, commands = self.step(5, llm_resp='FINAL: {"token":"placeholder_token"}')
        self.assertEqual("", command)
        self.assertEqual({}, commands)
        self.assertIn("包含占位值", prompt)

    def test_failed_query_does_not_count_as_evidence(self) -> None:
        self.step(2)
        self.step(3, last_cmd_result="[exitCode:0]\n./task_1_alpha.md")
        self.step(4, last_cmd_result="[exitCode:0]\nGET /api")
        self.step(5, llm_resp="CMD: curl -s http://127.0.0.1:1234/api")
        prompt, _, _ = self.step(6, last_cmd_result="[exitCode:1]\nconnection refused")
        self.assertFalse(self.memory.task_verified_tool_output)
        self.assertIn("没有成功返回有效数据", prompt)
        prompt, _, commands = self.step(7, llm_resp='FINAL: {"token":"guess"}')
        self.assertEqual({}, commands)
        self.assertIn("尚未取得成功的题目查询结果", prompt)

    def test_empty_query_output_does_not_count_as_evidence(self) -> None:
        self.step(2)
        self.step(3, last_cmd_result="[exitCode:0]\n./task_1_alpha.md")
        self.step(4, last_cmd_result="[exitCode:0]\nGET /api")
        self.step(5, llm_resp="CMD: curl -s http://127.0.0.1:1234/api")
        self.step(6, last_cmd_result="[exitCode:0]\n")
        self.assertFalse(self.memory.task_verified_tool_output)

    def test_more_placeholder_answers_are_blocked(self) -> None:
        for answer in ('{"token":"pending"}', '{"token":"waiting_for_check_output"}'):
            with self.subTest(answer=answer):
                memory = Memory(
                    task_state="exploring", task_started_round=2, task_point=Pos(14, 14),
                    task_desc=self.desc, task_steps_tried=3, task_llm_tool_calls=1,
                    pending_prompt_round=4, pending_prompt_kind="task",
                    task_verified_tool_output=True,
                )
                self.memory = memory
                _, _, commands = self.step(5, llm_resp=f"FINAL: {answer}")
                self.assertEqual({}, commands)
                self.assertEqual("", memory.task_answer)

    def test_task_deadline_limits_further_queries(self) -> None:
        task = replace(self.turn.player_tasks[0], position=Pos(14, 14), timeout_rounds=5)
        self.turn = replace(self.turn, player_tasks=(task,))
        self.memory = Memory(
            task_state="exploring", task_started_round=1, task_point=Pos(14, 14),
            task_desc=self.desc, task_steps_tried=2,
        )
        prompt, _, _ = self.step(4)
        self.assertIn("剩余回合不足以继续查询", prompt)
        _, command, commands = self.step(5, llm_resp="CMD: curl -s http://127.0.0.1/api")
        self.assertEqual("", command)
        self.assertEqual({}, commands)
        self.assertEqual("", self.memory.task_next_command)

    def test_night_does_not_move_task_pioneer(self) -> None:
        role = replace(self.role, pos=Pos(13, 14))
        station = self.turn.station()
        self.assertIsNotNone(station)
        turn = replace(self.turn, round_no=71, is_day=False, ours=(role, station))
        control_commands: dict = {}
        defense.night(turn, set(), control_commands)
        self.assertEqual("move", control_commands[role.unit_id]["action"])
        commands: dict = {}
        defense.night(turn, set(), commands, {role.unit_id})
        self.assertNotIn(role.unit_id, commands)


if __name__ == "__main__":
    unittest.main()

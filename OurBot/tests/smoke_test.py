#!/usr/bin/env python3
"""冒烟测试: 用 docs/request.txt 作为样例回合, 模拟跑满 1300 回合。

验证目标:
1. decide() 在昼/夜两种回合都不抛异常;
2. 指令 action 均为合法动作码, targetPos 均为 [{x,y}] 结构;
3. server 的 JSON 序列化可正常进行。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import protocol as P  # noqa: E402
from agent.brain import decide  # noqa: E402

VALID_ACTIONS = {
    "move", "attack", "sell", "buy", "build", "remove",
    "acceptTask", "submitAnswer", "summonTreasure", "use", "drop", "collect",
}


def main() -> None:
    sample = json.loads((ROOT.parent / "docs" / "request.txt").read_text(encoding="utf-8"))
    ok_rounds = 0
    total_cmds = 0
    problems: list[str] = []

    for round_no in range(1, P.ROUNDS_PER_DAY * P.TOTAL_DAYS + 1):
        sample["roundNo"] = round_no
        # 让机器人位置每回合轻微变化, 模拟动态场景
        for i, robot in enumerate(sample.get("robot", {}).get("roles", []) or []):
            robot["pos"]["x"] = (robot["pos"]["x"] + (i % 3) - 1) % P_WIDTH
        try:
            result = decide(sample)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"round {round_no}: decide raised {exc!r}")
            break

        cmd_map = result.get("roleCommandMap") or {}
        assert isinstance(cmd_map, dict)
        for role_id, cmd in cmd_map.items():
            total_cmds += 1
            action = cmd.get("action")
            if action not in VALID_ACTIONS:
                problems.append(f"round {round_no}: illegal action {action!r}")
            if "targetPos" in cmd:
                for pos in cmd["targetPos"]:
                    if not (isinstance(pos, dict) and "x" in pos and "y" in pos):
                        problems.append(f"round {round_no}: bad targetPos {pos!r}")
        # 响应序列化检查
        json.dumps(result, ensure_ascii=False)
        ok_rounds += 1

    print(f"rounds ok: {ok_rounds}/1300, commands issued: {total_cmds}")
    if problems:
        print("PROBLEMS:")
        for p in problems[:20]:
            print(" -", p)
        sys.exit(1)
    print("SMOKE TEST PASSED")


P_WIDTH = 41

if __name__ == "__main__":
    main()

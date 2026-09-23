"""任务引擎：自进化任务、LLM 流水线与长上下文宝藏。"""
import json
import logging
import re
import shlex
from dataclasses import replace
from typing import Any

from . import protocol as P
from .grid import adjacent_stands, path_to_any, next_step_adjacent
from .memory import Memory, SopRecord
from .protocol import (
    Pos,
    Unit,
    accept_task_command,
    buy_command,
    distance,
    move_command,
    submit_answer_command,
    summon_treasure_command,
)

LOGGER = logging.getLogger(__name__)

KNOWN_SHOP_ITEMS = {
    P.WEAPON_UP_V1, P.WEAPON_UP_V2,
    P.WALL_UP_V1, P.WALL_UP_V2,
    P.STATION_UP_V1, P.STATION_UP_V2,
    P.WALL_FIXER, P.MEDICINE, P.DIZZY, P.BOMB,
    P.SMALL_ORDER, P.MIDDLE_ORDER, P.LARGE_ORDER, P.BOSS_ORDER,
}
ACTIVE_TASK_STATES = {"accepting", "accepted", "exploring", "answering"}
MAX_LLM_TOOL_CALLS = 5
MIN_TASK_TIMEOUT = 10
ALLOWED_TASK_TOOLS = {
    "python", "python3", "curl", "wget", "sqlite3", "jq",
    "pwd", "ls", "find", "cat", "sed", "grep", "head", "tail",
    "awk", "cut", "sort", "uniq", "wc", "file", "stat",
    "sha256sum", "sha512sum", "sha1sum", "md5sum", "openssl", "base64", "printenv",
}


def pioneer(
    turn: P.Turn,
    pioneer_role: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow_new_task: bool = True,
    allow_treasure: bool = True,
    departure_round: int = 0,
) -> tuple[str, str]:
    """开拓者状态机，返回本回合的 (prompt, executeCmd)。"""
    memory.task_departure_round = departure_round
    if turn.llm_resp:
        _consume_llm_resp(turn, memory, turn.llm_resp)
    elif memory.pending_prompt_round and turn.round_no - memory.pending_prompt_round > 2:
        LOGGER.warning("LLM response timeout for %s", memory.pending_prompt_kind)
        memory.pending_prompt_round = 0
        memory.pending_prompt_kind = ""

    _sync_task_state(turn, pioneer_role, memory)

    if departure_round and turn.round_no >= departure_round:
        if (memory.task_state in ACTIVE_TASK_STATES and memory.task_answer
                and memory.task_state != "answering" and memory.task_point is not None
                and distance(pioneer_role.pos, memory.task_point) <= 1):
            commands[pioneer_role.unit_id] = submit_answer_command(memory.task_answer)
            memory.task_state = "answering"
            memory.task_answered_round = turn.round_no
        elif memory.task_state in ACTIVE_TASK_STATES:
            memory.record_task_outcome(False)
            LOGGER.info("task return deadline reached; return to defense")
            memory.reset_task()
        return "", ""

    if memory.task_state in ACTIVE_TASK_STATES:
        return _task_pipeline(turn, pioneer_role, memory, commands)

    handled, prompt = (_treasure_action(turn, pioneer_role, memory, claimed, commands)
                       if allow_treasure else (False, ""))
    if handled:
        return prompt, ""

    # prompt 与角色动作可同回合提交，不因赶往任务点而饿死宝藏推理链路。
    inference_prompt = _maybe_request_treasure_inference(turn, memory) if allow_treasure else ""

    if allow_new_task and turn.is_day:
        target = _pick_task_point(turn, pioneer_role, memory)
        if target is not None:
            if distance(pioneer_role.pos, target) <= 1:
                commands[pioneer_role.unit_id] = accept_task_command()
                memory.task_state = "accepting"
                memory.task_started_round = turn.round_no
                memory.task_point = target
                memory.clear_task_approach()
                LOGGER.info("requesting task at %s", target)
                return inference_prompt, ""
            # 途中不重新按距离打分选另一个任务点；固定目的地后，每回合只重算安全路径。
            memory.task_approach_point = target
            step = next_step_adjacent(turn, pioneer_role, target, reserved=claimed)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[pioneer_role.unit_id] = move_command(step)
                return inference_prompt, ""
            memory.clear_task_approach()
    else:
        memory.clear_task_approach()

    return inference_prompt, ""


def _sync_task_state(turn: P.Turn, role: Unit, memory: Memory) -> None:
    error_codes = {error.code for error in turn.errors}
    if 1 in error_codes:
        LOGGER.info("task timed out")
        if memory.task_state in ACTIVE_TASK_STATES:
            memory.record_task_outcome(False)
        memory.reset_task()
        return
    if 2 in error_codes and memory.task_state == "answering":
        reason = "; ".join(error.description for error in turn.errors if error.code == 2)
        _reject_task_answer(memory, reason or "答案错误或不完整")
    elif memory.task_state == "answering" and turn.last_action_results.get(role.unit_id) is False:
        _reject_task_answer(memory, "提交动作被判非法")

    if (
        memory.task_state == "answering"
        and not turn.phase_task
        and turn.round_no > memory.task_answered_round
        and not ({1, 2, 4} & error_codes)
        and role.health > 0
        and memory.task_point is not None
        and distance(role.pos, memory.task_point) <= 1
    ):
        # 只有提交后任务实际结束、且没有失败反馈，才将答案视为可复用经验。
        memory.add_sop(memory.task_desc, _extract_steps(memory), memory.task_answer[:2000])
        LOGGER.info("task completed after answer submission")
        memory.record_task_outcome(True)
        memory.reset_task()
        return

    if memory.task_state == "accepting":
        if turn.phase_task:
            memory.task_state = "accepted"
            memory.task_desc = turn.phase_task
            LOGGER.info("task accepted")
        elif turn.last_action_results.get(role.unit_id) is False:
            memory.reset_task()
        elif turn.round_no - memory.task_started_round > 1:
            memory.reset_task()
        return

    if memory.task_state in {"accepted", "exploring", "answering"}:
        if turn.phase_task:
            memory.task_desc = turn.phase_task
        elif turn.round_no > memory.task_started_round + 1:
            memory.reset_task()
            return

    if (
        memory.task_state == "answering"
        and turn.phase_task
        and turn.round_no - memory.task_answered_round > 1
        and 2 not in error_codes
    ):
        _reject_task_answer(memory, "提交后任务仍未结束，需复核答案")


def _reject_task_answer(memory: Memory, reason: str) -> None:
    answer = memory.task_answer.strip()
    if answer and answer not in memory.task_rejected_answers:
        memory.task_rejected_answers.append(answer)
    memory.task_transcript.append(
        f"上次提交未通过：{reason[:300]}。已尝试答案：{answer[:300]!r}。"
        "请继续核对沙盒文件，不要重复提交相同答案。"
    )
    LOGGER.warning("task answer rejected: %s; answer=%r", reason[:160], answer[:80])
    memory.task_state = "exploring"
    memory.task_answer = ""
    memory.task_answered_round = 0
    memory.pending_prompt_round = 0
    memory.pending_prompt_kind = ""


def _pick_task_point(
    turn: P.Turn,
    role: Unit | None = None,
    memory: Memory | None = None,
) -> Pos | None:
    ready = [task for task in turn.player_tasks if task.is_valid and task.cold_down == 0]
    if not ready:
        return None
    role = role or next(iter(turn.pioneers()), None)
    if role is None:
        return None
    scored = []
    for task in ready:
        path = path_to_any(turn, role, adjacent_stands(turn, role, (task.position,)))
        if path is None:
            continue
        travel = len(path)
        station = turn.station()
        return_steps = 0
        if station:
            arrived = replace(role, pos=path[-1] if path else role.pos)
            back = path_to_any(turn, arrived, adjacent_stands(turn, arrived, P.station_footprint(station.pos)))
            if back is None:
                continue
            return_steps = len(back)
        duration = task.timeout_rounds or 12
        # 实战中 10~12 回合任务来不及完成“查文件→查接口→LLM作答”的闭环，反复超时只会浪费昼间。
        if task.timeout_rounds and task.timeout_rounds < MIN_TASK_TIMEOUT:
            continue
        phase_round = (turn.round_no - 1) % P.ROUNDS_PER_DAY + 1
        if not turn.is_day:
            continue
        rounds_left = P.DAY_ROUNDS - phase_round
        if travel + duration + return_steps + 3 > rounds_left:
            continue
        wins, attempts = memory.task_outcomes.get(task.position, (0, 0)) if memory else (0, 0)
        probability = (wins + 1) / (attempts + 2)
        value = probability * (task.score_reward + task.gold_reward) / max(1, travel + duration)
        scored.append((value, task.position))
    if memory and memory.task_approach_point is not None:
        locked = next(
            (position for _, position in scored if position == memory.task_approach_point),
            None,
        )
        if locked is not None:
            return locked
        memory.clear_task_approach()
    return max(scored, key=lambda item: item[0])[1] if scored else None


def _task_pipeline(
    turn: P.Turn,
    role: Unit,
    memory: Memory,
    commands: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    task_pos = memory.task_point
    if task_pos is None:
        memory.reset_task()
        return "", ""
    if memory.task_state == "accepting":
        return "", ""
    if distance(role.pos, task_pos) > 1:
        memory.reset_task()
        return "", ""

    desc = memory.task_desc or turn.phase_task
    if not desc:
        return "", ""

    if memory.task_last_command_round:
        if turn.last_cmd_result and memory.task_last_result_round != turn.round_no:
            query_succeeded = _successful_query_result(turn.last_cmd_result)
            memory.task_transcript.append(
                f"命令{memory.task_steps_tried}: {memory.task_last_command}\n"
                f"结果:\n{turn.last_cmd_result[:16000]}"
            )
            if memory.task_last_command_is_query:
                if query_succeeded:
                    memory.task_verified_tool_output = True
                else:
                    memory.task_transcript.append(
                        "这条查询没有成功返回有效数据；不要据此猜答案，请换一种查询方式。"
                    )
            memory.task_last_result_round = turn.round_no
            memory.task_last_command_round = 0
            memory.task_last_command = ""
            memory.task_last_command_is_query = False
            LOGGER.info("task sandbox result received (success=%s, %d chars)",
                        query_succeeded, len(turn.last_cmd_result))
        elif turn.round_no - memory.task_last_command_round <= 1:
            return "", ""
        else:
            memory.task_transcript.append(
                f"命令{memory.task_steps_tried}: {memory.task_last_command}\n"
                "结果: 沙盒未返回结果，请改用其他查询方式。"
            )
            memory.task_last_command_round = 0
            memory.task_last_command = ""
            memory.task_last_command_is_query = False

    if memory.task_state == "accepted":
        # 即使任务描述相同，沙盒数据也可能变化；不能直接提交历史答案。
        memory.task_state = "exploring"

    if memory.task_state == "answering":
        return "", ""

    if memory.task_answer:
        commands[role.unit_id] = submit_answer_command(memory.task_answer)
        memory.task_state = "answering"
        memory.task_answered_round = turn.round_no
        LOGGER.info("task answer submitted (%d chars), awaiting result", len(memory.task_answer))
        return "", ""

    if memory.pending_prompt_kind == "task":
        return "", ""

    if memory.task_next_command:
        command = memory.task_next_command
        memory.task_next_command = ""
        memory.task_llm_tool_calls += 1
        return "", _issue_task_command(turn, memory, command, is_query=True)

    next_cmd = _next_explore_command(memory, desc)
    if next_cmd is not None:
        return "", _issue_task_command(turn, memory, next_cmd)

    return _request_task_llm(turn, memory, desc, memory.find_sop(desc)), ""


def _issue_task_command(
    turn: P.Turn, memory: Memory, command: str, *, is_query: bool = False,
) -> str:
    memory.task_steps_tried += 1
    memory.task_last_command = command
    memory.task_last_command_round = turn.round_no
    memory.task_last_command_is_query = is_query
    memory.task_command_history.append(command)
    LOGGER.info("task sandbox command issued (step %d, tool %s, %d chars)",
                memory.task_steps_tried, command.split(maxsplit=1)[0], len(command))
    return command


def _successful_query_result(result: str) -> bool:
    match = re.match(r"^\[exitCode:(-?\d+)\]\r?\n(.*)$", result, re.DOTALL)
    return bool(match and int(match.group(1)) == 0 and match.group(2).strip())


def _task_rounds_left(turn: P.Turn, memory: Memory) -> int | None:
    task = next((item for item in turn.player_tasks if item.position == memory.task_point), None)
    limits = []
    if task is not None and task.timeout_rounds > 0:
        limits.append(task.timeout_rounds - (turn.round_no - memory.task_started_round))
    if memory.task_departure_round:
        limits.append(memory.task_departure_round - turn.round_no)
    return max(0, min(limits)) if limits else None


def _tool_calls_left(turn: P.Turn, memory: Memory, *, responding: bool = False) -> int:
    left = max(0, MAX_LLM_TOOL_CALLS - memory.task_llm_tool_calls)
    rounds_left = _task_rounds_left(turn, memory)
    if rounds_left is None:
        return left
    # 发 prompt 后至少还需 CMD 回复、执行结果、FINAL 回复三个回合。
    reserve = 1 if responding else 2
    return min(left, max(0, (rounds_left - reserve) // 2))


def _next_explore_command(memory: Memory, desc: str) -> str | None:
    """只做最低限度的文档发现；后续命令由任务反馈驱动。"""
    step = memory.task_steps_tried
    local_target = _extract_local_target(desc)
    if step == 0:
        return "pwd; find . -maxdepth 4 -type f -print | head -100"
    if step == 1 and local_target:
        filename = local_target.rsplit("/", 1)[-1]
        return (
            f"find . -maxdepth 5 -type f -name '{filename}' "
            "-print -exec sed -n '1,240p' {} \\; 2>/dev/null | head -300"
        )
    if step <= 1:
        return (
            "find . -maxdepth 4 -type f "
            "\\( -iname '*.md' -o -iname '*.txt' -o -iname '*.json' \\) "
            "-print -exec sed -n '1,160p' {} \\; 2>/dev/null | head -1200"
        )
    return None


def _extract_local_target(desc: str) -> str:
    match = re.search(
        r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_./-]+\.(?:md|txt|json|yaml|yml))"
        r"(?=$|[^A-Za-z0-9_./-])",
        desc,
    )
    return match.group(1) if match else ""


def _request_task_llm(
    turn: P.Turn,
    memory: Memory,
    desc: str,
    sop: SopRecord | None,
) -> str:
    if memory.pending_prompt_round:
        return ""
    memory.pending_prompt_round = turn.round_no
    memory.pending_prompt_kind = "task"
    transcript = "\n\n".join(memory.task_transcript)[-24000:] or "（无沙盒输出）"
    prior = ""
    if sop is not None:
        prior = (
            "\n--- 相似任务经验（仅供选择查询步骤，答案可能已过期） ---\n"
            f"任务: {sop.fingerprint}\n曾用命令: {sop.steps[:5]}\n"
        )
    rejected = "\n".join(memory.task_rejected_answers[-3:])
    if rejected:
        prior += f"\n--- 已被判错误或未完成的答案（禁止原样重复） ---\n{rejected}\n"
    available_calls = _tool_calls_left(turn, memory)
    if available_calls == 0:
        next_action = "剩余回合不足以继续查询；立即 FINAL 提交已核实字段以争取部分分，缺少证据的字段省略，禁止猜测。"
    elif not memory.task_verified_tool_output:
        next_action = "尚未取得成功的题目查询结果，本次必须先给出一条沙盒查询命令。"
    else:
        next_action = "优先 FINAL 提交已经核实的字段保底得分，再根据判题反馈查询和补齐缺失字段。不要等所有字段齐全才首次提交。"
    return (
        "你是游戏内任务求解器。沙盒可执行基础 shell/Python 命令且无外网。"
        "先依据任务文档实际查询 API、数据文件或计算 token；不能猜测数值，也不能用 unknown/placeholder 充数。"
        "每次只返回一行，二选一：CMD: <一条只读 shell 命令> 或 FINAL: <提交答案>。"
        "规则按正确字段给部分积分和金币；FINAL 可为包含已核实字段的部分答案，保持题目规定的结构。"
        "不要解释、不要 Markdown 围栏。CMD 会由 executeCmd 执行，结果在下一次消息中给你。"
        "命令不要修改/删除文件、不要访问外网、不要用分号或重定向；"
        "可用至多两段只读命令通过 | 连接，例如 curl ... | jq ...。"
        "URL 含 & 时须用引号包住。复杂计算可用 python3 -c；哈希可用 sha256sum 等工具。"
        f"{next_action}最多还能执行 {available_calls} 条自选命令。\n"
        f"--- 当前任务 ---\n{desc}\n"
        f"--- 沙盒探索记录 ---\n{transcript}\n"
        f"{prior}下一步:"
    )


def _extract_steps(memory: Memory) -> list[str]:
    return list(memory.task_command_history)


def _parse_task_llm_output(resp: str) -> tuple[str, str]:
    value = resp.strip()
    fenced = re.fullmatch(r"```(?:json|text|bash|sh)?\s*\n(.*?)\n```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        kind = str(parsed.get("kind") or parsed.get("type") or "").lower()
        if kind in {"command", "cmd", "executecmd"}:
            return "command", str(parsed.get("command") or parsed.get("cmd") or "").strip()
        if kind in {"answer", "final"}:
            answer = parsed.get("answer", "")
            return "answer", answer.strip() if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return "answer", json.dumps(parsed, ensure_ascii=False)
    match = re.match(r"^(?:CMD|COMMAND|EXECUTECMD)\s*[:：]\s*(.*)$", value, re.IGNORECASE | re.DOTALL)
    if match:
        return "command", match.group(1).strip()
    match = re.match(r"^(?:FINAL|ANSWER)\s*[:：]\s*(.*)$", value, re.IGNORECASE | re.DOTALL)
    if match:
        return "answer", match.group(1).strip()
    return "answer", value


def _safe_task_command(command: str) -> bool:
    """只允许最多两段只读查询管道，禁止连接符与重定向。"""
    if (not command or len(command) > 2000 or "\n" in command or "\r" in command
            or "$(" in command or "`" in command):
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>`")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return False
    if not tokens or any(token in {";", "&", "&&", "||", ">", ">>", "<", "<<", "`"}
                         for token in tokens):
        return False
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token == "|":
            segments.append([])
        elif re.fullmatch(r"[;&|<>`]+", token):
            return False
        else:
            segments[-1].append(token)
    return (len(segments) <= 2 and all(
        segment and segment[0].lower() in ALLOWED_TASK_TOOLS
        for segment in segments
    ))


def _answer_has_placeholder(answer: str) -> bool:
    try:
        value = json.loads(answer)
    except (ValueError, TypeError):
        value = answer

    def is_placeholder(item: Any) -> bool:
        if isinstance(item, dict):
            return any(is_placeholder(part) for part in item.values())
        if isinstance(item, list):
            return any(is_placeholder(part) for part in item)
        if not isinstance(item, str):
            return False
        text = item.strip().lower()
        return (text in {"unknown", "todo", "tbd", "xxx", "null", "pending", "n/a"}
                or text.startswith(("placeholder", "waiting_for_")))

    return is_placeholder(value)


def _consume_llm_resp(turn: P.Turn, memory: Memory, resp: str) -> None:
    if not memory.pending_prompt_round:
        return
    kind = memory.pending_prompt_kind
    if kind == "task" and memory.task_state in ACTIVE_TASK_STATES:
        response_kind, value = _parse_task_llm_output(resp)
        if response_kind == "command":
            if _tool_calls_left(turn, memory, responding=True) == 0:
                memory.task_transcript.append("查询额度或剩余回合不足；请依据已有结果尽快作答。")
            elif value in memory.task_command_history:
                memory.task_transcript.append("这条命令已经执行过，请换一种查询方式。")
            elif _safe_task_command(value):
                memory.task_next_command = value
                LOGGER.info("LLM requested task sandbox command (%d chars)", len(value))
            else:
                memory.task_transcript.append(
                    "命令格式不安全或不受支持；请使用允许的只读工具，最多两段管道，"
                    "不要使用分号、重定向或外网。"
                )
                LOGGER.warning("LLM requested unsupported task command (tool=%r, len=%d)",
                               value.split(maxsplit=1)[0] if value else "", len(value))
        elif not value or value in memory.task_rejected_answers or _answer_has_placeholder(value):
            memory.task_transcript.append("答案为空、包含占位值或此前已被判错；请继续查询证据。")
            LOGGER.warning("LLM returned empty, placeholder, or previously rejected task answer")
        elif not memory.task_verified_tool_output:
            memory.task_transcript.append("尚未取得成功的题目查询结果；请先给出 CMD，不要猜最终答案。")
            LOGGER.warning("LLM attempted task answer before querying sandbox")
        else:
            memory.task_answer = value
            LOGGER.info("LLM produced task answer (%d chars)", len(value))
        memory.pending_prompt_round = 0
        memory.pending_prompt_kind = ""
    elif kind == "treasure":
        memory.llm_consumed(P.day_index(turn.round_no))
        _parse_treasure_answer(turn, memory, resp)


def task_supply_names(turn: P.Turn) -> tuple[str, ...]:
    """商店中除固定升级/消耗品外的条目均视为本地图任务用品。"""
    return tuple(name for name in turn.shop_prices if name and name not in KNOWN_SHOP_ITEMS)


def treasure_plan_prompt(turn: P.Turn, memory: Memory) -> str:
    lines = [f"DAY{d}: {text}" for d, text in sorted(memory.folk_legends.items())]
    supplies = ", ".join(task_supply_names(turn)) or "（商店未提供候选）"
    return (
        "根据连续多日民间传闻推断唯一宝藏。只能使用候选物品名，信息不足时明确输出 UNKNOWN。\n"
        "严格按三行格式输出：\n位置: (x, y)\n物品: name1,name2\n时间: DAYn 白天/夜晚"
        "（若有精确回合范围也写出）\n"
        f"候选任务用品: {supplies}\n\n" + "\n".join(lines)
    )


def _maybe_request_treasure_inference(turn: P.Turn, memory: Memory) -> str:
    if memory.treasure_done or memory.pending_prompt_round or len(memory.folk_legends) < 2:
        return ""
    day = P.day_index(turn.round_no)
    legend_count = len(memory.folk_legends)
    if (
        memory.treasure_last_prompt_legend_count == legend_count
        or memory.treasure_last_prompt_day == day
        or memory.llm_budget_left(day) <= 0
    ):
        return ""
    memory.pending_prompt_round = turn.round_no
    memory.pending_prompt_kind = "treasure"
    memory.treasure_last_prompt_day = day
    memory.treasure_last_prompt_legend_count = legend_count
    return treasure_plan_prompt(turn, memory)


def _parse_treasure_answer(turn: P.Turn, memory: Memory, resp: str) -> None:
    available = task_supply_names(turn)
    items = [name for name in available if name.lower() in resp.lower()]
    pos_match = re.search(r"[（(]\s*(\d+)\s*[,，]\s*(\d+)\s*[)）]", resp)
    time_match = re.search(r"时间\s*[:：]\s*(.+)", resp, re.IGNORECASE)
    if pos_match:
        memory.treasure_pos = Pos(int(pos_match.group(1)), int(pos_match.group(2)))
    if items:
        memory.treasure_items = items
    if time_match:
        memory.treasure_open_hint = time_match.group(1).strip()
    LOGGER.info(
        "treasure inference pos=%s items=%s time=%s",
        memory.treasure_pos,
        memory.treasure_items,
        memory.treasure_open_hint,
    )


def _treasure_action(
    turn: P.Turn,
    role: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> tuple[bool, str]:
    if memory.treasure_done:
        return False, ""
    items = memory.treasure_items or []
    pos = memory.treasure_pos
    if not items or pos is None or not memory.treasure_open_hint:
        return False, ""
    if _treasure_day_is_far(turn, memory.treasure_open_hint):
        return False, ""

    missing = next((item for item in items if not role.has(item)), None)
    if missing is not None:
        shop = turn.weapon_shop_pos()
        price = turn.shop_prices.get(missing, 0)
        station = turn.station()
        reserve = (100 if station.level == 1 else 150) if (
            station and station.level < 3 and station.health < 1500 * station.level * 0.65
        ) else 20
        if (
            shop is None
            or price <= 0
            or turn.gold - _reserved_gold(turn, commands) - reserve < price
            or role.backpack_full
        ):
            return False, ""  # 买不起宝藏用品时继续做能挣钱的任务。
        if distance(role.pos, shop) <= 1:
            commands[role.unit_id] = buy_command(missing, 1)
        else:
            step = next_step_adjacent(turn, role, shop, reserved=claimed)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[role.unit_id] = move_command(step)
        return True, ""

    if distance(role.pos, pos) > 1:
        step = next_step_adjacent(turn, role, pos, reserved=claimed)
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[role.unit_id] = move_command(step)
        return True, ""

    if _treasure_time_matches(turn, memory.treasure_open_hint) and not memory.treasure_pending_round:
        commands[role.unit_id] = summon_treasure_command(pos, items)
        memory.treasure_pending_round = turn.round_no
    return True, ""


def _reserved_gold(turn: P.Turn, commands: dict[int, dict[str, Any]]) -> int:
    total = 0
    for command in commands.values():
        if command.get("action") == "build" and command.get("name") in P.TOWER_TYPES:
            total += P.WEAPON_BUILD_COST
        elif command.get("action") == "buy":
            total += turn.shop_prices.get(str(command.get("name") or ""), 0) * int(
                command.get("num") or 1
            )
    return total


def _treasure_day_is_far(turn: P.Turn, hint: str) -> bool:
    hinted_day = _hinted_day(hint)
    return hinted_day is not None and hinted_day > P.day_index(turn.round_no) + 1


def _treasure_time_matches(turn: P.Turn, hint: str) -> bool:
    normalized = hint.lower()
    hinted_day = _hinted_day(normalized)
    if hinted_day is not None and hinted_day != P.day_index(turn.round_no):
        return False
    round_range = re.search(
        r"(?:回合|round)\s*(\d+)\s*[-~至到]\s*(\d+)"
        r"|第?\s*(\d+)\s*[-~至到]\s*(\d+)\s*回合",
        normalized,
    )
    if round_range:
        values = [int(value) for value in round_range.groups() if value is not None]
        if not values[0] <= turn.round_no <= values[1]:
            return False
    if any(word in normalized for word in ("夜晚", "黑夜", "night")) and turn.is_day:
        return False
    if any(word in normalized for word in ("白天", "daytime")) and not turn.is_day:
        return False
    return bool(hinted_day is not None or round_range or any(
        word in normalized for word in ("夜晚", "黑夜", "night", "白天", "daytime")
    ))


def _hinted_day(hint: str) -> int | None:
    match = re.search(r"day\s*(\d+)|第\s*(\d+)\s*天", hint, re.IGNORECASE)
    if not match:
        return None
    return int(next(value for value in match.groups() if value is not None))


def handle_summon_result(turn: P.Turn, memory: Memory) -> None:
    code = turn.last_summon_result
    if not memory.treasure_pending_round and code == 0:
        return
    memory.treasure_pending_round = 0
    if code in (1, 4):
        memory.treasure_done = True
        LOGGER.info("treasure is no longer available (result=%d)", code)
    elif code == 3:
        memory.treasure_items = None
        memory.treasure_last_prompt_legend_count = 0
        LOGGER.warning("treasure items were wrong; scheduling re-inference")
    elif code == 2:
        memory.treasure_open_hint = ""
        memory.treasure_last_prompt_legend_count = 0
        LOGGER.warning("treasure not open at inferred place/time")

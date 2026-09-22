"""任务引擎：自进化任务、LLM 流水线与长上下文宝藏。"""
import logging
import re
from typing import Any

from . import protocol as P
from .grid import next_step_adjacent
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


def pioneer(
    turn: P.Turn,
    pioneer_role: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow_new_task: bool = True,
) -> tuple[str, str]:
    """开拓者状态机，返回本回合的 (prompt, executeCmd)。"""
    if turn.llm_resp:
        _consume_llm_resp(turn, memory, turn.llm_resp)
    elif memory.pending_prompt_round and turn.round_no - memory.pending_prompt_round > 2:
        LOGGER.warning("LLM response timeout for %s", memory.pending_prompt_kind)
        memory.pending_prompt_round = 0
        memory.pending_prompt_kind = ""

    _sync_task_state(turn, pioneer_role, memory)

    if memory.task_state in ACTIVE_TASK_STATES:
        return _task_pipeline(turn, pioneer_role, memory, commands)

    handled, prompt = _treasure_action(turn, pioneer_role, memory, claimed, commands)
    if handled:
        return prompt, ""

    # prompt 与角色动作可同回合提交，不因赶往任务点而饿死宝藏推理链路。
    inference_prompt = _maybe_request_treasure_inference(turn, memory)

    if allow_new_task and turn.is_day:
        target = _pick_task_point(turn)
        if target is not None:
            if distance(pioneer_role.pos, target) <= 1:
                commands[pioneer_role.unit_id] = accept_task_command()
                memory.task_state = "accepting"
                memory.task_started_round = turn.round_no
                memory.task_point = target
                LOGGER.info("requesting task at %s", target)
                return inference_prompt, ""
            step = next_step_adjacent(turn, pioneer_role, target)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[pioneer_role.unit_id] = move_command(step)
                return inference_prompt, ""

    return inference_prompt, ""


def _sync_task_state(turn: P.Turn, role: Unit, memory: Memory) -> None:
    error_codes = {error.code for error in turn.errors}
    if 1 in error_codes:
        LOGGER.info("task timed out")
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


def _pick_task_point(turn: P.Turn) -> Pos | None:
    ready = [task for task in turn.player_tasks if task.is_valid and task.cold_down == 0]
    if not ready:
        return None
    return max(
        ready,
        key=lambda task: (task.score_reward + task.gold_reward, task.timeout_rounds),
    ).position


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

    if memory.task_steps_tried and turn.last_cmd_result and memory.task_last_result_round != turn.round_no:
        memory.task_transcript.append(
            f"步骤{memory.task_steps_tried}输出:\n{turn.last_cmd_result[:12000]}"
        )
        memory.task_last_result_round = turn.round_no

    if memory.task_state == "accepted":
        sop = memory.find_sop(desc)
        if sop is not None and sop.fingerprint == desc and sop.answer_hint:
            memory.task_answer = sop.answer_hint
        # 相似任务的答案不能直接用于当前任务；先读取当前沙盒材料。
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

    next_cmd = _next_explore_command(memory, desc)
    if next_cmd is not None:
        memory.task_steps_tried += 1
        return "", next_cmd

    return _request_task_llm(turn, memory, desc, memory.find_sop(desc)), ""


def _next_explore_command(memory: Memory, desc: str) -> str | None:
    """先发现任务文件，再读取候选文档；所有输出都会累计进 prompt。"""
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
    if not memory.task_rejected_answers:
        return None
    if step == 2 and local_target:
        filename = local_target.rsplit("/", 1)[-1]
        return (
            f"find . -maxdepth 5 -type f -name '{filename}' "
            "-print -exec sed -n '241,800p' {} \\; 2>/dev/null | head -600"
        )
    if step <= 3:
        return (
            "find . -maxdepth 5 -type f "
            "\\( -iname '*.md' -o -iname '*.txt' -o -iname '*.json' "
            "-o -iname '*.yaml' -o -iname '*.yml' \\) "
            "-print -exec sed -n '1,240p' {} \\; 2>/dev/null | head -1200"
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
            "\n--- 相似任务经验（只作参考，必须按当前参数改写） ---\n"
            f"任务: {sop.fingerprint}\n答案: {sop.answer_hint}\n"
        )
    rejected = "\n".join(memory.task_rejected_answers[-3:])
    if rejected:
        prior += f"\n--- 已被判错误或未完成的答案（禁止原样重复） ---\n{rejected}\n"
    return (
        "你是游戏内任务求解器。请根据当前任务、沙盒输出和相似任务经验，直接输出可提交的完整答案。"
        "不要解释，不要添加 Markdown 围栏；必须覆盖任务要求的所有字段。\n"
        f"--- 当前任务 ---\n{desc}\n"
        f"--- 沙盒探索记录 ---\n{transcript}\n"
        f"{prior}最终答案:"
    )


def _extract_steps(memory: Memory) -> list[str]:
    return [entry.split("输出:", 1)[0] for entry in memory.task_transcript]


def _consume_llm_resp(turn: P.Turn, memory: Memory, resp: str) -> None:
    if not memory.pending_prompt_round:
        return
    kind = memory.pending_prompt_kind
    if kind == "task" and memory.task_state in ACTIVE_TASK_STATES:
        answer = resp.strip()
        if answer and answer not in memory.task_rejected_answers:
            memory.task_answer = answer
            LOGGER.info("LLM produced task answer (%d chars)", len(answer))
        else:
            LOGGER.warning("LLM returned empty or previously rejected task answer")
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
        if (
            shop is None
            or price <= 0
            or turn.gold - _reserved_gold(turn, commands) < price
            or role.backpack_full
        ):
            return True, ""
        if distance(role.pos, shop) <= 1:
            commands[role.unit_id] = buy_command(missing, 1)
        else:
            step = next_step_adjacent(turn, role, shop)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[role.unit_id] = move_command(step)
        return True, ""

    if distance(role.pos, pos) > 1:
        step = next_step_adjacent(turn, role, pos)
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

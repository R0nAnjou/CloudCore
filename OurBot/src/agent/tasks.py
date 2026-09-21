"""任务引擎: 自进化任务(acceptTask -> executeCmd 循环 -> submitAnswer)
+ LLM 流水线(本回合填 prompt, 下回合读 llmResp)
+ 长上下文宝藏(民间传闻 -> 献祭召唤)。

关键约束(接口文档):
- prompt 每游戏日限 3 次; 自进化任务执行期间不受限。
- executeCmd 本回合提交, 下回合从 lastCmdResult 拿结果, 15 秒超时, 仅任务期可用。
- summonTreasure 只要动作合法就消耗物品, 必须确认三要素后再献祭。
"""
import logging
import re
from typing import Any

from . import protocol as P
from .grid import next_step
from .memory import Memory
from .protocol import (
    Pos,
    Unit,
    accept_task_command,
    distance,
    move_command,
    submit_answer_command,
    summon_treasure_command,
)

LOGGER = logging.getLogger(__name__)

# 常见任务用品名(价值 15 金), 背包采购候选
TASK_SUPPLIES = (
    "AcientTablet",
    "StarSand",
    "FlameBreath",
    "FrostPotion",
    "ThornAmulet",
    "IronWhistle",
)


def pioneer(
    turn: P.Turn,
    pioneer_role: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    """开拓者白天状态机。返回 (prompt, executeCmd) 供响应体填充。"""
    prompt = ""
    execute_cmd = ""

    # 0) 响应 LLM 结果(若有)
    if turn.llm_resp:
        _consume_llm_resp(turn, memory, turn.llm_resp)

    # 1) 已在任务执行期 -> 走自进化流水线
    if memory.task_state in ("accepted", "exploring") and memory.task_point is not None:
        prompt, execute_cmd = _task_pipeline(turn, pioneer_role, memory, claimed, commands)
        return prompt, execute_cmd

    # 2) 尝试领任务: 任务点有效且开拓者能到
    if memory.task_state == "idle":
        target = _pick_task_point(turn)
        if target is not None:
            if distance(pioneer_role.pos, target) <= 1:
                commands[pioneer_role.unit_id] = accept_task_command()
                memory.task_state = "accepted"
                memory.task_started_round = turn.round_no
                memory.task_point = target
                memory.task_desc = turn.phase_task
                LOGGER.info("task accepted at %s", target)
                return prompt, execute_cmd
            step = next_step(turn, pioneer_role, target)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[pioneer_role.unit_id] = move_command(step)
                return prompt, execute_cmd

    # 3) 空闲: 去武器商店买任务用品(为宝藏做准备), 否则待命
    _idle_shopping(turn, pioneer_role, memory, claimed, commands)
    return prompt, execute_cmd


def _pick_task_point(turn: P.Turn) -> Pos | None:
    for task in turn.player_tasks:
        if task.is_valid and task.cold_down == 0:
            return task.position
    return None


def _task_pipeline(
    turn: P.Turn,
    role: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    """自进化任务三阶段: SOP 命中 -> 直接作答; 未命中 -> 探索。"""
    prompt = ""
    execute_cmd = ""

    # 保持站在任务点旁(离开则任务结束!)
    task_pos = memory.task_point
    assert task_pos is not None
    if distance(role.pos, task_pos) > 1:
        step = next_step(turn, role, task_pos)
        if step is not None and step not in claimed:
            claimed.add(step)
            commands[role.unit_id] = move_command(step)
        return prompt, execute_cmd

    desc = memory.task_desc or turn.phase_task or ""
    if not desc:
        # 尚未拿到任务原文, 先领一次
        commands[role.unit_id] = accept_task_command()
        return prompt, execute_cmd

    # 阶段A: SOP 命中 -> 直接提交(0 次 LLM)
    sop = memory.find_sop(desc)
    if sop is not None and memory.task_state == "accepted":
        answer = _render_answer(sop, desc)
        commands[role.unit_id] = submit_answer_command(answer)
        memory.task_state = "answered"
        LOGGER.info("SOP hit, submitted directly")
        return prompt, execute_cmd

    # 阶段B: 探索 -> 本回合提交 executeCmd, 下回合读 lastCmdResult
    if memory.task_state == "accepted":
        memory.task_state = "exploring"
        memory.task_steps_tried = 0

    next_cmd = _next_explore_command(memory, desc, turn.last_cmd_result)
    if next_cmd is not None:
        execute_cmd = next_cmd
        memory.task_steps_tried += 1
        return prompt, execute_cmd

    # 阶段C: 探索完成 -> 组装答案提交; 必要时请求 LLM 归纳(任务期不限额)
    if memory.task_answer:
        commands[role.unit_id] = submit_answer_command(memory.task_answer)
        memory.task_state = "answered"
        memory.add_sop(desc[:60], _extract_steps(memory), memory.task_answer[:200])
        LOGGER.info("task answered (round %d)", turn.round_no)
        return prompt, execute_cmd

    # 需要 LLM 归纳 -> 发 prompt
    if not memory.pending_prompt_round:
        memory.pending_prompt_round = turn.round_no
        prompt = _build_task_prompt(memory, desc, turn.last_cmd_result)
    return prompt, execute_cmd


def _next_explore_command(memory: Memory, desc: str, last_result: str) -> str | None:
    """按探索进度生成下一条沙盒命令; 返回 None 表示探索完成。

    首版策略: 用固定的 5 步探路脚本摸清 API 文档与调用方式。
    """
    step_index = memory.task_steps_tried
    # 从任务描述中提取 URL 或模块名作为探索目标
    m = re.search(r"(https?://[^\s\"']+|[\w./-]+api[\w./-]*)", desc)
    target = m.group(1) if m else ""
    script: list[str] = []
    if last_result is None:
        last_result = ""
    if step_index == 0:
        script = ["ls -la; pwd"]
    elif step_index == 1:
        script = ["find / -maxdepth 2 -iname '*api*' -o -iname '*doc*' 2>/dev/null | head -20"]
    elif step_index == 2 and target:
        script = [f"cat {target} 2>/dev/null | head -100" if not target.startswith("http") else f"echo DOC_URL:{target}"]
    elif step_index == 3 and target:
        script = [f"python3 - <<'EOF'\nimport json,urllib.request\nprint('exploring {target}')\nEOF"]
    elif step_index == 4:
        script = ["echo PROBE_DONE"]
    else:
        return None
    return script[0] if script else None


def _extract_steps(memory: Memory) -> list[str]:
    # 首版: 记录我们尝试过的命令序列作为 SOP 步骤
    return [
        "ls -la; pwd",
        "find / -maxdepth 2 -iname '*api*' -o -iname '*doc*'",
        "cat <doc>",
        "python probe",
    ]


def _render_answer(sop: Any, desc: str) -> str:
    """SOP 命中后的答案渲染(首版: 返回 SOP 提示 + 通用答案骨架)。"""
    base = sop.answer_hint or ""
    return base or "SOP:" + sop.fingerprint


def _build_task_prompt(memory: Memory, desc: str, last_result: str) -> str:
    """让 LLM 根据任务描述+沙盒输出, 产出最终答案。"""
    return (
        "你是游戏内的任务求解器。以下是任务描述与沙盒探索输出, 请直接给出任务要求的最终答案, "
        "不要解释。若无法确定, 给出最可能的答案并标注 [GUESS]。\n"
        f"--- 任务描述 ---\n{desc}\n"
        f"--- 沙盒输出(截断) ---\n{last_result[:4000]}\n"
        f"--- 已尝试步骤 ---\n{memory.task_steps_tried}\n"
        "最终答案:"
    )


def _consume_llm_resp(turn: P.Turn, memory: Memory, resp: str) -> None:
    """下回合收到 llmResp: 存为任务答案/宝藏线索。"""
    if memory.pending_prompt_round:
        memory.llm_consumed(P.day_index(turn.round_no))
        if memory.task_state in ("accepted", "exploring"):
            memory.task_answer = resp.strip()
            LOGGER.info("LLM produced task answer (%d chars)", len(resp))
        else:
            # 宝藏解谜: 尝试解析出物品与坐标
            _parse_treasure_answer(memory, resp)
        memory.pending_prompt_round = 0


def _parse_treasure_answer(memory: Memory, resp: str) -> None:
    items = [name for name in TASK_SUPPLIES if name in resp]
    pos_m = re.search(r"\((\d+)\s*,\s*(\d+)\)", resp)
    if items:
        memory.treasure_items = items[:3]
    if pos_m:
        memory.treasure_pos = Pos(int(pos_m.group(1)), int(pos_m.group(2)))


# ---- 长上下文宝藏 ------------------------------------------------------
def treasure_plan_prompt(memory: Memory) -> str:
    """民间传闻累计 N 天后, 请求 LLM 推断宝藏三要素。"""
    lines = [f"DAY{d}: {text}" for d, text in sorted(memory.folk_legends.items())]
    return (
        "以下是连续多日的民间传闻。请推断祭坛宝藏的三要素并严格按格式输出:\n"
        "位置: (x, y)\n物品: <任务用品英文名列表, 逗号分隔, 最多3个>\n时间: <开启窗口的描述>\n\n"
        + "\n".join(lines)
    )


def try_summon_treasure(
    turn: P.Turn,
    role: Unit,
    memory: Memory,
    commands: dict[int, dict[str, Any]],
) -> bool:
    """若三要素齐备则尝试召唤。返回是否发出指令。"""
    items = memory.treasure_items or []
    pos = memory.treasure_pos
    if not items or pos is None:
        return False
    if not all(role.has(item) for item in items):
        return False
    if (pos.x, pos.y) in memory.treasure_tried_pos:
        return False
    if distance(role.pos, pos) > 1:
        return False  # 位置移动由 pioneer 状态机处理
    commands[role.unit_id] = summon_treasure_command(pos, items)
    memory.treasure_tried_pos.add((pos.x, pos.y))
    return True


def handle_summon_result(turn: P.Turn, memory: Memory) -> None:
    """根据 lastSummonTreasureResult 在线纠错。"""
    code = turn.last_summon_result
    if code == 3:
        # 献祭物品错 -> 清空让 LLM 重新推断
        memory.treasure_items = None
        LOGGER.warning("summonTreasure: wrong items, will re-infer")
    elif code == 2:
        # 无宝藏/未到时间 -> 换位置重试
        if memory.treasure_pos is not None:
            memory.treasure_tried_pos.add((memory.treasure_pos.x, memory.treasure_pos.y))


def _idle_shopping(
    turn: P.Turn,
    role: Unit,
    memory: Memory,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """空闲开拓者: 去武器商店补齐任务用品(每样1件, 15金/件)。"""
    shop = turn.weapon_shop_pos()
    if shop is None:
        return
    need = next((s for s in TASK_SUPPLIES if not role.has(s)), None)
    if need is None or turn.gold < 15 or role.backpack_full:
        return
    if distance(role.pos, shop) <= 1:
        from .protocol import buy_command

        commands[role.unit_id] = buy_command(need, 1)
        return
    step = next_step(turn, role, shop)
    if step is not None and step not in claimed:
        claimed.add(step)
        commands[role.unit_id] = move_command(step)

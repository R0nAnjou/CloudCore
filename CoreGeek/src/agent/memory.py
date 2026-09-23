"""跨回合记忆: 建造区试错学习、新闻/传闻日志、SOP 知识库、宝藏线索。

本模块是唯一持有全局可变状态的地方, server 进程存活期间持续有效。
"""
import logging
import re
from dataclasses import dataclass, field

from .protocol import Pos

LOGGER = logging.getLogger(__name__)

# 矿石关键词 -> 矿名
_ORE_KEYWORDS = {
    "石": "stone",
    "铁": "iron",
    "铜": "copper",
    "stone": "stone",
    "iron": "iron",
    "copper": "copper",
}
# 停产事件关键词
_BAD_NEWS = ("塌方", "停工", "事故", "检修", "停产", "掩埋", "枯竭", "禁采")


@dataclass
class SopRecord:
    """一条可复用的技能(SOP): 任务描述特征 -> 命令序列模板。"""

    fingerprint: str
    steps: list[str] = field(default_factory=list)
    answer_hint: str = ""


@dataclass
class Memory:
    """全局跨回合状态。"""

    # 1) 建造区试错学习: (pos, kind) -> 最近一次 build 是否合法
    build_ok: dict[tuple[int, int, str], bool] = field(default_factory=dict)
    build_retry_after: dict[tuple[int, int, str], int] = field(default_factory=dict)
    build_failures: dict[tuple[int, int, str], int] = field(default_factory=dict)
    current_round: int = 0
    worker_modes: dict[int, str] = field(default_factory=dict)
    returning_roles: set[int] = field(default_factory=set)
    gate_pos: Pos | None = None
    tower_layout: tuple[Pos, ...] = ()
    task_outcomes: dict[Pos, tuple[int, int]] = field(default_factory=dict)
    # 2) 每日官方新闻存档 {day: text}
    official_news: dict[int, str] = field(default_factory=dict)
    # 3) 每日民间传闻存档 {day: text}
    folk_legends: dict[int, str] = field(default_factory=dict)
    # 4) 新闻推断的矿产波动: ore -> {day: affected}
    ore_outage: dict[str, set[int]] = field(default_factory=dict)
    # 5) 宝藏解谜状态
    treasure_items: list[str] | None = None
    treasure_pos: Pos | None = None
    treasure_open_hint: str = ""
    treasure_done: bool = False
    treasure_pending_round: int = 0
    treasure_last_prompt_day: int = 0
    treasure_last_prompt_legend_count: int = 0
    # 6) LLM 流水线状态
    pending_prompt_round: int = 0  # 发出 prompt 的回合号, 0=无
    pending_prompt_kind: str = ""  # task / treasure
    llm_calls_this_day: int = 0
    llm_day: int = 0
    # 7) 自进化任务状态机
    task_state: str = "idle"  # idle / accepting / accepted / exploring / answering
    task_started_round: int = 0
    task_departure_round: int = 0
    task_point: Pos | None = None
    task_desc: str = ""
    task_steps_tried: int = 0
    task_answer: str = ""
    task_answered_round: int = 0
    task_rejected_answers: list[str] = field(default_factory=list)
    task_transcript: list[str] = field(default_factory=list)
    task_last_result_round: int = 0
    task_next_command: str = ""
    task_last_command: str = ""
    task_last_command_round: int = 0
    task_last_command_is_query: bool = False
    task_command_history: list[str] = field(default_factory=list)
    task_llm_tool_calls: int = 0
    task_verified_tool_output: bool = False
    # 8) SOP 知识库
    sops: list[SopRecord] = field(default_factory=list)
    # 9) 上回合发出的指令缓存(兜底重发用)
    last_commands: dict[str, dict] = field(default_factory=dict)
    # 10) 调试统计
    rounds_seen: int = 0

    # ------------------------------------------------------------------
    def remember_news(self, day: int, official: str, folk: str) -> None:
        if official and self.official_news.get(day) != official:
            self.official_news[day] = official
            self._parse_official(day, official)
            LOGGER.info("news day%d official: %s", day, official[:80])
        if folk and self.folk_legends.get(day) != folk:
            self.folk_legends[day] = folk
            LOGGER.info("news day%d folk: %s", day, folk[:80])

    def _parse_official(self, day: int, text: str) -> None:
        """推理类任务: 从官方消息推断哪种矿在未来几天停产(价格将上涨)。"""
        for keyword, ore in _ORE_KEYWORDS.items():
            if keyword in text and any(bad in text for bad in _BAD_NEWS):
                # 假设: 当天可采, 明后两天停产(任务书 5.1 示例口径)
                affected = self.ore_outage.setdefault(ore, set())
                affected.update({day + 1, day + 2})
                LOGGER.warning("ore outage predicted: %s on days %s", ore, affected)

    def ore_blocked(self, ore: str, day: int) -> bool:
        return day in self.ore_outage.get(ore, set())

    # ------------------------------------------------------------------
    def note_build_result(self, pos: Pos, kind: str, ok: bool) -> None:
        key = (pos.x, pos.y, kind)
        if ok:
            self.build_failures.pop(key, None)
            self.build_retry_after.pop(key, None)
        else:
            # 接口只给成功/失败，不能把临时占位或资金不足当成永久非法。
            failures = self.build_failures.get(key, 0) + 1
            self.build_failures[key] = failures
            self.build_retry_after[key] = self.current_round + min(20, 2 ** min(failures, 5))

    def build_allowed(self, pos: Pos, kind: str) -> bool:
        key = (pos.x, pos.y, kind)
        return (self.build_ok.get(key, True)
                and self.current_round >= self.build_retry_after.get(key, 0))

    def record_task_outcome(self, completed: bool) -> None:
        if self.task_point is not None:
            wins, attempts = self.task_outcomes.get(self.task_point, (0, 0))
            self.task_outcomes[self.task_point] = (wins + int(completed), attempts + 1)

    # ------------------------------------------------------------------
    def llm_budget_left(self, day: int) -> int:
        if self.llm_day != day:
            return 3
        return max(0, 3 - self.llm_calls_this_day)

    def llm_consumed(self, day: int) -> None:
        if self.llm_day != day:
            self.llm_day = day
            self.llm_calls_this_day = 0
        self.llm_calls_this_day += 1
        self.pending_prompt_round = 0
        self.pending_prompt_kind = ""

    def reset_task(self) -> None:
        self.task_state = "idle"
        self.task_started_round = 0
        self.task_point = None
        self.task_desc = ""
        self.task_steps_tried = 0
        self.task_answer = ""
        self.task_answered_round = 0
        self.task_rejected_answers.clear()
        self.task_transcript.clear()
        self.task_last_result_round = 0
        self.task_next_command = ""
        self.task_last_command = ""
        self.task_last_command_round = 0
        self.task_last_command_is_query = False
        self.task_command_history.clear()
        self.task_llm_tool_calls = 0
        self.task_verified_tool_output = False
        if self.pending_prompt_kind == "task":
            self.pending_prompt_round = 0
            self.pending_prompt_kind = ""

    # ------------------------------------------------------------------
    def find_sop(self, task_desc: str) -> SopRecord | None:
        """按任务描述的粗特征指纹查 SOP(首版: 关键词重合度)。"""
        best: SopRecord | None = None
        best_score = 0
        words = _task_tokens(task_desc)
        for sop in self.sops:
            key_words = _task_tokens(sop.fingerprint)
            union = words | key_words
            score = len(words & key_words) / len(union) if union else 0
            if score > best_score:
                best, best_score = sop, score
        return best if best_score >= 0.55 else None

    def add_sop(self, fingerprint: str, steps: list[str], answer_hint: str = "") -> None:
        record = SopRecord(fingerprint=fingerprint, steps=list(steps), answer_hint=answer_hint)
        self.sops.append(record)
        LOGGER.info("SOP saved: %s (%d steps)", fingerprint[:40], len(steps))


def _task_tokens(text: str) -> set[str]:
    """英文按词、中文按双字组生成稳定指纹，避免整句中文只成为一个 token。"""
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    tokens.update(chinese[i:i + 2] for i in range(max(0, len(chinese) - 1)))
    return tokens

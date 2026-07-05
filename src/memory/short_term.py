# src/memory/short_term.py
"""
短时记忆 (Short-Term Memory)
=============================
单个会话内的高效记忆管理，提供:

1. **滑动窗口** — 保留最近 N 条消息，超出时自动截断
2. **Token 感知截断** — 按 token 预算而非消息条数截断，适配不同 LLM 上下文窗口
3. **溢出摘要** — 被截断的旧消息自动压缩为摘要，保留关键信息
4. **重要信息标记** — 支持标记和提取关键决策（目的地确认、偏好变更等）

设计原则:
    - 全内存操作，不涉及 I/O（会话结束即消失，或归档到 LTM）
    - 与 TravelState.messages 协同工作，不重复存储
    - Token 计数使用轻量估算，不引入额外依赖
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.memory.types import MemoryEntry

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """会话级短时记忆。

    使用示例:
        stm = ShortTermMemory(max_tokens=4000)
        stm.add("user", "我想去北京玩三天")
        stm.add("assistant", "好的，正在为您搜索北京景点...")
        stm.add("agent", "找到15个景点", metadata={"agent": "attraction"})

        # 获取上下文（自动截断 + 摘要）
        context = stm.get_context(max_tokens=2000)
    """

    # ── 默认配置 ──────────────────────────────────────────────────────────

    DEFAULT_MAX_MESSAGES = 40  # 消息条数硬上限
    DEFAULT_MAX_TOKENS = 8000  # token 软上限（触发摘要）
    SUMMARY_TRIGGER_RATIO = 0.8  # 使用率达到此比例时触发摘要
    SUMMARY_KEEP_RECENT = 8  # 摘要后始终保留最近 N 条完整消息
    SUMMARY_MIN_OLD = 6  # 至少有这么多条旧消息才触发摘要

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_messages: int = DEFAULT_MAX_MESSAGES,
    ):
        self.max_tokens = max_tokens
        self.max_messages = max_messages

        # 核心存储
        self._messages: List[MemoryEntry] = []
        self._summary: str = ""  # 被截断消息的压缩摘要
        self._key_decisions: List[Dict[str, Any]] = []  # 关键决策记录

        # 统计
        self._total_messages_added: int = 0

    # ── 公共 API ──────────────────────────────────────────────────────────

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """添加一条消息到短时记忆。

        Args:
            role: 消息角色 (user / assistant / agent / system)
            content: 消息文本
            metadata: 附加信息 (agent_name, importance 等)

        Returns:
            创建的 MemoryEntry
        """
        entry = MemoryEntry(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(entry)
        self._total_messages_added += 1

        # 检查是否需要压缩
        if self._should_summarize():
            self._summarize_old_messages()

        # 硬截断保护
        if len(self._messages) > self.max_messages:
            overflow = self._messages[: -self.max_messages]
            self._messages = self._messages[-self.max_messages :]
            logger.debug(f"硬截断: 丢弃 {len(overflow)} 条消息")

        return entry

    def mark_decision(self, key: str, value: Any, note: str = "") -> None:
        """标记一个关键决策（目的地确认、偏好变更等）。

        这些决策在摘要时会被优先保留。
        """
        self._key_decisions.append(
            {
                "key": key,
                "value": value,
                "note": note,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.debug(f"关键决策: {key} = {value}")

    def get_context(
        self,
        max_tokens: Optional[int] = None,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """获取当前会话上下文。

        Args:
            max_tokens: 返回上下文的最大 token 数（None = 使用默认值）
            include_summary: 是否包含溢出摘要

        Returns:
            {
                "messages": [{"role": "...", "content": "..."}, ...],
                "summary": "...",           # 溢出摘要（如有）
                "key_decisions": [...],     # 关键决策列表
                "estimated_tokens": int,    # 估算总 token 数
            }
        """
        budget = max_tokens or self.max_tokens

        # 从最新消息开始收集，直到接近 token 预算
        selected: List[MemoryEntry] = []
        token_sum = 0

        for entry in reversed(self._messages):
            t = entry.estimated_tokens
            if token_sum + t > budget:
                break
            selected.append(entry)
            token_sum += t

        selected.reverse()  # 恢复时间顺序

        result: Dict[str, Any] = {
            "messages": [m.to_message_dict() for m in selected],
            "key_decisions": list(self._key_decisions),
            "estimated_tokens": token_sum,
        }

        if include_summary and self._summary:
            result["summary"] = self._summary

        return result

    def get_all_messages(self) -> List[MemoryEntry]:
        """返回所有消息（不做截断）。"""
        return list(self._messages)

    def clear(self) -> None:
        """清空当前会话记忆。"""
        self._messages.clear()
        self._summary = ""
        self._key_decisions.clear()

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _should_summarize(self) -> bool:
        """判断是否需要触发摘要压缩。"""
        total_tokens = sum(m.estimated_tokens for m in self._messages)
        trigger_threshold = int(self.max_tokens * self.SUMMARY_TRIGGER_RATIO)
        old_count = len(self._messages) - self.SUMMARY_KEEP_RECENT
        return (
            total_tokens > trigger_threshold and old_count >= self.SUMMARY_MIN_OLD
        )

    def _summarize_old_messages(self) -> None:
        """将旧消息压缩为摘要，释放 token 空间。

        当前实现使用规则式提取（不调用 LLM）:
        - 保留最近的 KEEP_RECENT 条消息不变
        - 对旧消息提取: 用户原始意图、确认的目的地/天数、关键景点名
        - 生成简洁的结构化摘要

        未来可升级为 LLM 驱动的摘要。
        """
        if len(self._messages) <= self.SUMMARY_KEEP_RECENT:
            return

        split_idx = len(self._messages) - self.SUMMARY_KEEP_RECENT
        old_messages = self._messages[:split_idx]
        self._messages = self._messages[split_idx:]

        # ── 规则式摘要提取 ────────────────────────────────────────────
        extracted_parts: List[str] = []

        # 1. 提取用户原始查询
        user_msgs = [m for m in old_messages if m.role == "user"]
        if user_msgs:
            extracted_parts.append(f"用户查询: {user_msgs[0].content[:200]}")
            if len(user_msgs) > 1:
                extracted_parts.append(
                    f"后续追问: {user_msgs[-1].content[:150]}"
                )

        # 2. 提取 Agent 决策
        agent_msgs = [m for m in old_messages if m.role == "agent"]
        for m in agent_msgs:
            agent_name = m.metadata.get("agent", m.metadata.get("agent_name", ""))
            if agent_name:
                extracted_parts.append(
                    f"[{agent_name}] {m.content[:120]}"
                )

        # 3. 提取关键决策
        if self._key_decisions:
            decisions_str = "; ".join(
                f"{d['key']}={d['value']}" for d in self._key_decisions[-5:]
            )
            extracted_parts.append(f"关键决策: {decisions_str}")

        # 4. 合并已有摘要（如果有）
        if self._summary:
            extracted_parts.insert(0, f"[上轮摘要] {self._summary[:300]}")

        self._summary = " | ".join(extracted_parts) if extracted_parts else ""
        logger.debug(
            f"摘要压缩: {len(old_messages)} 条旧消息 → "
            f"{len(self._summary)} 字符摘要 "
            f"(保留最近 {len(self._messages)} 条)"
        )

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def total_tokens(self) -> int:
        return sum(m.estimated_tokens for m in self._messages)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def key_decisions(self) -> List[Dict[str, Any]]:
        return list(self._key_decisions)

    def __repr__(self) -> str:
        return (
            f"<ShortTermMemory msgs={len(self._messages)} "
            f"tokens≈{self.total_tokens} summary={len(self._summary)}chars>"
        )

# src/memory/context.py
"""
上下文管理器 (Context Manager)
================================
负责在 Agent 执行前，将短时记忆、长时记忆和当前状态融合为"即用型上下文"。

核心职责:
    1. **多源融合** — 从 STM、LTM、TravelState 中提取相关信息
    2. **Token 预算** — 按优先级分配 token，确保不超出 LLM 上下文窗口
    3. **优先级排序** — 当前输入 > 用户画像 > 最近对话 > 相似历史
    4. **压缩回退** — 超预算时自动截断低优先级内容

使用示例:
    cm = ContextManager(stm, ltm, token_budget=6000)
    ctx = await cm.assemble(
        user_id="u1",
        session_id="s1",
        current_state={"user_input": "北京三日游", "destination": "北京"},
    )
    # ctx.to_prompt_fragment() → 可直接拼入 LLM prompt
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.memory.types import AssembledContext
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)


class ContextManager:
    """上下文组装器。

    按优先级组装上下文:
        P0 (必选): 当前用户输入
        P1 (高优先): 用户长期画像
        P2 (中优先): 短时记忆中的最近消息
        P3 (低优先): 语义检索到的相似历史记忆
    """

    # ── Token 预算分配比例 ────────────────────────────────────────────────
    BUDGET_P0_CURRENT_INPUT = 0.10   # 当前输入占 10%（通常很短）
    BUDGET_P1_USER_PROFILE = 0.15    # 用户画像占 15%
    BUDGET_P2_RECENT_MSGS = 0.45     # 最近消息占 45%
    BUDGET_P3_SIMILAR_MEMORIES = 0.25  # 相似历史占 25%
    BUDGET_RESERVE = 0.05            # 5% 预留

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        token_budget: int = 6000,
    ):
        self.short_term = short_term
        self.long_term = long_term
        self.token_budget = token_budget

    # ── 主入口 ────────────────────────────────────────────────────────────

    async def assemble(
        self,
        user_id: str,
        session_id: str,
        current_state: Dict[str, Any],
        agent_name: str = "supervisor",
    ) -> AssembledContext:
        """组装当前 Agent 所需的完整上下文。

        Args:
            user_id: 用户标识
            session_id: 会话标识
            current_state: 当前 TravelState 字典
            agent_name: 调用方 Agent 名称（用于日志和针对性优化）

        Returns:
            AssembledContext — 可直接调用 .to_prompt_fragment() 拼入 prompt
        """
        budget = self.token_budget
        sources: List[str] = []

        # ── P0: 当前用户输入 ──────────────────────────────────────────
        p0_budget = int(budget * self.BUDGET_P0_CURRENT_INPUT)
        current_input = current_state.get("user_input", "")
        if len(current_input) > p0_budget * 4:  # 粗略 char→token
            current_input = current_input[: p0_budget * 4] + "..."
        sources.append("current_input")

        # ── P1: 用户画像 ──────────────────────────────────────────────
        p1_budget = int(budget * self.BUDGET_P1_USER_PROFILE)
        user_profile = await self.long_term.get_user_profile(user_id)
        profile_text = user_profile.to_context_string()
        if len(profile_text) > p1_budget * 4:
            # 截断profile中较长的部分
            lines = profile_text.split("\n")
            truncated = []
            char_count = 0
            for line in lines:
                if char_count + len(line) > p1_budget * 4:
                    break
                truncated.append(line)
                char_count += len(line)
            profile_text = "\n".join(truncated)
        if profile_text:
            sources.append("user_profile")

        # ── P2: 短时记忆（最近消息） ──────────────────────────────────
        p2_budget = int(budget * self.BUDGET_P2_RECENT_MSGS)
        stm_context = self.short_term.get_context(
            max_tokens=p2_budget, include_summary=True
        )
        recent_messages = stm_context.get("messages", [])
        summary = stm_context.get("summary", "")
        if recent_messages:
            sources.append(f"recent_{len(recent_messages)}_messages")
        if summary:
            sources.append("stm_summary")

        # ── P3: 相似历史记忆 ──────────────────────────────────────────
        p3_budget = min(
            int(budget * self.BUDGET_P3_SIMILAR_MEMORIES),
            1500,  # 硬上限，避免检索结果占用过多
        )
        search_query = self._build_search_query(current_state)
        similar_memories = await self.long_term.retrieve_similar(
            query=search_query,
            user_id=user_id,
            limit=self.long_term.config.max_sessions_to_retrieve,
        )
        # 截断相似记忆到 token 预算
        truncated_memories: List[Dict[str, Any]] = []
        mem_tokens = 0
        for mem in similar_memories:
            t = len(mem.get("content", "")) // 4
            if mem_tokens + t > p3_budget:
                break
            truncated_memories.append(mem)
            mem_tokens += t
        if truncated_memories:
            sources.append(f"similar_{len(truncated_memories)}_memories")

        # ── 构建系统上下文文本 ────────────────────────────────────────
        system_parts = []
        if current_input:
            system_parts.append(f"[当前任务]\n用户请求: {current_input}")

        # 附加当前状态的关键信息
        dest = current_state.get("destination", "")
        num_days = current_state.get("num_days", 0)
        has_attr = bool(current_state.get("attractions"))
        has_it = bool(current_state.get("itinerary"))
        if dest:
            parts = [f"目的地: {dest}"]
            if num_days:
                parts.append(f"天数: {num_days}")
            parts.append(f"景点: {'已就绪' if has_attr else '待搜索'}")
            parts.append(f"行程: {'已生成' if has_it else '待生成'}")
            system_parts.append(" | ".join(parts))

        if profile_text:
            system_parts.append(profile_text)

        if summary:
            system_parts.append(f"[会话摘要] {summary}")

        system_context = "\n\n".join(system_parts)

        # ── 计算总 token ──────────────────────────────────────────────
        total_tokens = (
            len(system_context) // 4
            + sum(len(m.get("content", "")) // 4 for m in recent_messages)
            + sum(len(m.get("content", "")) // 4 for m in truncated_memories)
        )

        logger.debug(
            f"上下文组装完成: agent={agent_name} "
            f"sources={sources} tokens≈{total_tokens}/{budget}"
        )

        return AssembledContext(
            system_context=system_context,
            recent_messages=recent_messages,
            relevant_memories=truncated_memories,
            token_count=total_tokens,
            source_labels=sources,
        )

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    def _build_search_query(self, state: Dict[str, Any]) -> str:
        """从当前状态构建用于语义检索的查询文本。"""
        parts = []
        destination = state.get("destination", "")
        user_input = state.get("user_input", "")
        interests = state.get("interests", [])

        if destination:
            parts.append(destination)
        if user_input:
            parts.append(user_input)
        if interests:
            parts.append(" ".join(interests))

        return " ".join(parts) if parts else user_input

    def update_budget(self, new_budget: int) -> None:
        """动态调整 token 预算。"""
        self.token_budget = max(500, new_budget)
        logger.debug(f"Token 预算已更新: {self.token_budget}")

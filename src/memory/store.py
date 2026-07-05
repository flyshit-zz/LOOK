# src/memory/store.py
"""
记忆存储门面 (Memory Store)
=============================
统一对外接口，管理短时记忆、长时记忆和上下文管理器的生命周期。

上层（Agent / Graph / FastAPI）只需与 MemoryStore 交互，无需关心底层:
    - 短时记忆自动管理（随会话创建/销毁）
    - 长时记忆自动归档（会话结束时持久化）
    - 上下文自动组装（Agent 执行前调用）

典型用法:
    # 应用启动
    store = MemoryStore()
    await store.initialize()

    # 新会话开始
    session = store.create_session(session_id="s1", user_id="u1")

    # Agent 执行前获取上下文
    ctx = await store.get_context(user_id="u1", session_id="s1", state={...})

    # 记录 Agent 产出
    store.record_agent_message(session_id="s1", agent="attraction", content="...")

    # 会话结束归档
    await store.archive_session(session_id="s1", state=final_state)

    # 应用关闭
    await store.close()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.memory.types import (
    AssembledContext,
    LongTermMemoryConfig,
    SessionSummary,
    UserProfile,
)
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.context import ContextManager

logger = logging.getLogger(__name__)


class MemoryStore:
    """记忆模块统一门面。

    属性:
        long_term (LongTermMemory): 长时记忆实例（跨会话持久化）
        config (LongTermMemoryConfig): 配置

    内部:
        _sessions (Dict[str, ShortTermMemory]): 活跃会话的短时记忆映射
        _context_manager (ContextManager): 上下文组装器
    """

    # ── 默认配置 ──────────────────────────────────────────────────────────

    DEFAULT_TOKEN_BUDGET = 6000  # 上下文窗口 token 预算
    DEFAULT_STM_MAX_TOKENS = 4000  # 单会话短时记忆 token 上限

    def __init__(
        self,
        config: Optional[LongTermMemoryConfig] = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ):
        self.config = config or LongTermMemoryConfig()
        self.token_budget = token_budget

        # 核心组件（延迟初始化）
        self.long_term: Optional[LongTermMemory] = None
        self._context_manager: Optional[ContextManager] = None

        # 活跃会话的短时记忆
        self._sessions: Dict[str, ShortTermMemory] = {}

        # 用户记忆缓存
        self._user_profiles: Dict[str, UserProfile] = {}

    # ── 生命周期 ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """初始化记忆存储（应用启动时调用一次）。"""
        self.long_term = LongTermMemory(config=self.config)
        await self.long_term.initialize()
        logger.info(
            f"MemoryStore 已初始化: token_budget={self.token_budget}"
        )

    async def close(self) -> None:
        """关闭记忆存储，释放资源。"""
        # 归档所有活跃会话
        for sid, stm in list(self._sessions.items()):
            if stm.message_count > 0:
                logger.warning(f"关闭时仍有活跃会话: {sid}（消息未归档）")
            self._sessions.pop(sid, None)

        if self.long_term:
            await self.long_term.close()

        self._user_profiles.clear()
        logger.info("MemoryStore 已关闭")

    # ── 会话管理 ──────────────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        user_id: str = "default",
        initial_user_input: str = "",
    ) -> ShortTermMemory:
        """为新会话创建短时记忆空间。

        Args:
            session_id: 会话唯一标识
            user_id: 用户标识
            initial_user_input: 初始用户输入（可选，立即记录）

        Returns:
            新创建的 ShortTermMemory 实例
        """
        if session_id in self._sessions:
            logger.debug(f"会话 {session_id} 已存在，返回现有实例")
            return self._sessions[session_id]

        stm = ShortTermMemory(
            max_tokens=self.DEFAULT_STM_MAX_TOKENS,
        )

        # 记录初始输入
        if initial_user_input:
            stm.add("user", initial_user_input)

        self._sessions[session_id] = stm
        logger.info(f"会话已创建: {session_id} (user={user_id})")
        return stm

    def get_session(self, session_id: str) -> Optional[ShortTermMemory]:
        """获取活跃会话的短时记忆。"""
        return self._sessions.get(session_id)

    def ensure_session(
        self, session_id: str, user_id: str = "default"
    ) -> ShortTermMemory:
        """获取或创建会话。"""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(session_id, user_id)

    # ── 消息记录 ──────────────────────────────────────────────────────────

    def record_user_message(
        self, session_id: str, content: str
    ) -> None:
        """记录用户消息。"""
        stm = self.ensure_session(session_id)
        stm.add("user", content)

    def record_agent_message(
        self,
        session_id: str,
        agent: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录 Agent 产出消息。

        Args:
            session_id: 会话标识
            agent: Agent 名称 (supervisor / attraction / itinerary)
            content: 消息内容（进度提示、决策说明等）
            metadata: 附加元数据
        """
        stm = self.ensure_session(session_id)
        meta = metadata or {}
        meta["agent"] = agent
        stm.add("agent", content, metadata=meta)

    def record_decision(
        self, session_id: str, key: str, value: Any, note: str = ""
    ) -> None:
        """记录关键决策（目的地确认、偏好变更等）。"""
        stm = self.ensure_session(session_id)
        stm.mark_decision(key, value, note)

    # ── 上下文获取 ────────────────────────────────────────────────────────

    async def get_context(
        self,
        user_id: str,
        session_id: str,
        state: Dict[str, Any],
        agent_name: str = "supervisor",
    ) -> AssembledContext:
        """获取 Agent 执行所需上下文。

        这是 Agent 执行前的主入口 —— 融合 STM + LTM + 当前状态。

        Args:
            user_id: 用户标识
            session_id: 会话标识
            state: 当前 TravelState（来自 LangGraph）
            agent_name: 调用方 Agent 名称

        Returns:
            AssembledContext（含 system_context, recent_messages,
            relevant_memories 等）
        """
        if not self.long_term:
            raise RuntimeError("MemoryStore 未初始化，请先调用 initialize()")

        stm = self.ensure_session(session_id, user_id)

        if self._context_manager is None:
            self._context_manager = ContextManager(
                short_term=stm,
                long_term=self.long_term,
                token_budget=self.token_budget,
            )
        else:
            # 更新 context manager 使用的 stm 实例（可能已切换会话）
            self._context_manager.short_term = stm

        ctx = await self._context_manager.assemble(
            user_id=user_id,
            session_id=session_id,
            current_state=state,
            agent_name=agent_name,
        )

        logger.debug(
            f"上下文已组装: agent={agent_name} "
            f"sources={ctx.source_labels} tokens≈{ctx.token_count}"
        )
        return ctx

    # ── 用户画像 ──────────────────────────────────────────────────────────

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """获取用户画像（带缓存）。"""
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]

        if not self.long_term:
            raise RuntimeError("MemoryStore 未初始化")

        profile = await self.long_term.get_user_profile(user_id)
        self._user_profiles[user_id] = profile
        return profile

    async def update_user_preferences(
        self, user_id: str, preferences: Dict[str, Any]
    ) -> UserProfile:
        """更新用户偏好并刷新缓存。"""
        if not self.long_term:
            raise RuntimeError("MemoryStore 未初始化")

        profile = await self.long_term.update_preferences(user_id, preferences)
        self._user_profiles[user_id] = profile
        return profile

    # ── 会话归档 ──────────────────────────────────────────────────────────

    async def archive_session(
        self, session_id: str, state: Dict[str, Any], user_id: str = "default"
    ) -> Optional[SessionSummary]:
        """归档已完成的会话。

        将会话内容持久化到长时记忆:
        1. 提取偏好并更新用户画像
        2. 生成 SessionSummary 存入 SQLite + ChromaDB
        3. 清理短时记忆

        Args:
            session_id: 会话标识
            state: 最终 TravelState
            user_id: 用户标识

        Returns:
            SessionSummary（或 None，如果会话不存在或未完成）
        """
        stm = self._sessions.get(session_id)
        if stm is None:
            logger.warning(f"归档时找不到会话: {session_id}")
            return None

        if not self.long_term:
            raise RuntimeError("MemoryStore 未初始化")

        destination = state.get("destination", "")
        itinerary = state.get("itinerary", "")
        daily_routes = state.get("daily_routes", [])
        user_input = state.get("user_input", "")
        num_days = state.get("num_days", 0)

        # ── Step 1: 提取用户偏好并更新画像 ────────────────────────────
        extracted_prefs = self._extract_preferences_from_state(state)
        if extracted_prefs:
            await self.long_term.update_preferences(user_id, extracted_prefs)
            # 刷新缓存
            self._user_profiles.pop(user_id, None)

        # ── Step 2: 提取景点名称 ──────────────────────────────────────
        attraction_names = []
        if daily_routes:
            for day in daily_routes:
                for attr in day.get("attractions", []):
                    name = attr.get("name", "")
                    if name:
                        attraction_names.append(name)

        # ── Step 3: 创建会话摘要 ──────────────────────────────────────
        from datetime import datetime, timezone

        summary = SessionSummary(
            session_id=session_id,
            user_id=user_id,
            destination=destination,
            num_days=num_days,
            query=user_input,
            itinerary_preview=itinerary[:500] if itinerary else "",
            attraction_names=attraction_names[:20],
            timestamp=datetime.now(timezone.utc),
            metadata={
                "attraction_count": len(attraction_names),
                "itinerary_length": len(itinerary),
                "message_count": stm.message_count if stm else 0,
            },
        )

        await self.long_term.save_session_summary(summary)

        # ── Step 4: 清理短时记忆 ──────────────────────────────────────
        self._sessions.pop(session_id, None)

        logger.info(
            f"会话已归档: session={session_id} dest={destination} "
            f"attrs={len(attraction_names)}"
        )
        return summary

    # ── 偏好提取 ──────────────────────────────────────────────────────────

    def _extract_preferences_from_state(
        self, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """从 TravelState 中提取可持久化的用户偏好。"""
        prefs = {}

        # 兴趣标签
        interests = state.get("interests", [])
        if interests:
            prefs["interests"] = interests

        # 旅行风格
        travel_style = state.get("travel_style")
        if travel_style:
            prefs["travel_style"] = travel_style

        # 饮食偏好
        cuisine = state.get("cuisine_preference")
        if cuisine:
            prefs["cuisine_preference"] = cuisine

        # 住宿偏好
        accommodation = state.get("accommodation_preference")
        if accommodation:
            prefs["accommodation_preference"] = accommodation

        # 预算
        budget = state.get("budget")
        if budget:
            prefs["budget_range"] = str(budget)

        return prefs if prefs else {}

    # ── 状态查询 ──────────────────────────────────────────────────────────

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        return (
            f"<MemoryStore sessions={len(self._sessions)} "
            f"budget={self.token_budget}>"
        )

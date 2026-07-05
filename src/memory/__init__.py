# src/memory/__init__.py
"""
记忆模块 —— 为多 Agent 系统提供短时记忆、长时记忆和上下文管理。

架构概览:
    ShortTermMemory  —— 会话内滑动窗口 + 溢出摘要
    LongTermMemory   —— 跨会话 SQLite 结构化存储 + ChromaDB 向量检索
    ContextManager   —— 融合 STM/LTM，按 token 预算组装上下文
    MemoryStore      —— 统一门面，管理生命周期，对上层透明

典型用法:
    from src.memory import MemoryStore

    store = MemoryStore()
    await store.initialize()

    # Agent 获取上下文
    ctx = await store.get_context(user_id="u1", session_id="s1", current_state={...})

    # 会话结束后归档
    await store.archive_session(session_id="s1", state=final_state)
"""

from src.memory.embeddings import DashScopeEmbeddingFunction
from src.memory.types import (
    MemoryEntry,
    UserProfile,
    SessionSummary,
    AssembledContext,
    LongTermMemoryConfig,
)
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.context import ContextManager
from src.memory.store import MemoryStore

__all__ = [
    # 数据模型
    "MemoryEntry",
    "UserProfile",
    "SessionSummary",
    "AssembledContext",
    "LongTermMemoryConfig",
    # 核心组件
    "ShortTermMemory",
    "LongTermMemory",
    "ContextManager",
    "MemoryStore",
    # Embedding
    "DashScopeEmbeddingFunction",
]

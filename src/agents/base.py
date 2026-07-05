# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """Agent 基类。

    所有 Agent 继承此类，实现 execute 方法。
    memory_store 为可选注入 —— 当不为 None 时，Agent 可利用记忆模块
    获取用户画像、历史记忆和会话上下文。
    """

    def __init__(
        self,
        name: str,
        description: str,
        memory_store: Optional[Any] = None,
    ):
        self.name = name
        self.description = description
        self.memory_store = memory_store  # MemoryStore 实例（可选注入）

    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def get_memory_context(
        self, state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """从 MemoryStore 获取当前上下文（便捷方法）。

        子类在执行前调用此方法，可获取融合了用户画像、历史记忆
        和最近对话的组装上下文。
        """
        if self.memory_store is None:
            return None

        user_id = state.get("user_id", "default")
        session_id = state.get("session_id", "default")

        try:
            from src.memory.types import AssembledContext

            ctx: AssembledContext = await self.memory_store.get_context(
                user_id=user_id,
                session_id=session_id,
                state=state,
                agent_name=self.name,
            )
            return {
                "system_context": ctx.system_context,
                "recent_messages": ctx.recent_messages,
                "relevant_memories": ctx.relevant_memories,
                "token_count": ctx.token_count,
                "source_labels": ctx.source_labels,
            }
        except Exception:
            return None
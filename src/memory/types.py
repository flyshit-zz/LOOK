# src/memory/types.py
"""
记忆模块的数据类型定义。

所有 Pydantic 模型均支持序列化/反序列化，可直接存入 SQLite (JSON 字段)
或 ChromaDB (metadata + document)。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# 基础记忆条目
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryEntry(BaseModel):
    """单条记忆 —— 短时记忆的基本单位，也可持久化为长时记忆。"""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: str = Field(description="消息角色: user / assistant / system / agent")
    content: str = Field(description="记忆内容文本")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="创建时间 (UTC)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="附加元数据 (agent_name, token_count, importance 等)",
    )

    def to_message_dict(self) -> Dict[str, str]:
        """转为 LLM 可消费的消息字典格式。"""
        role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        return {"role": role_map.get(self.role, "user"), "content": self.content}

    @property
    def estimated_tokens(self) -> int:
        """粗略估算 token 数（中文 ~1.5 char/token，英文 ~4 char/token）。"""
        chars = len(self.content)
        # 中文字符占比估算
        cjk_chars = sum(1 for c in self.content if "一" <= c <= "鿿")
        other_chars = chars - cjk_chars
        return int(cjk_chars / 1.5 + other_chars / 4)


# ═══════════════════════════════════════════════════════════════════════════════
# 用户画像（长时记忆 — 结构化）
# ═══════════════════════════════════════════════════════════════════════════════

class UserProfile(BaseModel):
    """用户长期画像 —— 跨会话持久化，随每次交互逐步更新。"""

    user_id: str = Field(description="用户唯一标识")
    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "偏好字典，可包含: interests (List[str]), travel_style (str), "
            "cuisine_preference (str), accommodation_preference (str), "
            "budget_range (str), preferred_seasons (List[str]) 等"
        ),
    )
    past_destinations: List[str] = Field(
        default_factory=list, description="去过的目的地城市列表"
    )
    past_trip_count: int = Field(default=0, description="历史行程总数")
    favorite_attraction_types: List[str] = Field(
        default_factory=list, description="偏好的景点类型 (如 ['博物馆', '自然风光'])"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def merge_preferences(self, new_prefs: Dict[str, Any]) -> None:
        """合并新的偏好数据（非空字段覆盖，列表字段合并去重）。"""
        for key, value in new_prefs.items():
            if value is None or (isinstance(value, (list, str, dict)) and not value):
                continue
            existing = self.preferences.get(key)
            if isinstance(existing, list) and isinstance(value, list):
                self.preferences[key] = list(dict.fromkeys(existing + value))
            elif isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
            else:
                self.preferences[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def to_context_string(self) -> str:
        """将用户画像转为一段可供 LLM 阅读的上下文文本。"""
        parts = ["[用户长期偏好]"]

        prefs = self.preferences
        if prefs.get("interests"):
            parts.append(f"兴趣: {', '.join(prefs['interests'])}")
        if prefs.get("travel_style"):
            parts.append(f"旅行风格: {prefs['travel_style']}")
        if prefs.get("cuisine_preference"):
            parts.append(f"饮食偏好: {prefs['cuisine_preference']}")
        if prefs.get("budget_range"):
            parts.append(f"预算范围: {prefs['budget_range']}")

        if self.past_destinations:
            parts.append(f"去过: {', '.join(self.past_destinations[-5:])}")

        if self.favorite_attraction_types:
            parts.append(f"偏好景点类型: {', '.join(self.favorite_attraction_types)}")

        return "\n".join(parts) if len(parts) > 1 else ""


# ═══════════════════════════════════════════════════════════════════════════════
# 会话摘要（长时记忆 — 归档用）
# ═══════════════════════════════════════════════════════════════════════════════

class SessionSummary(BaseModel):
    """单次会话完成后归档的摘要，用于未来相似查询时检索。"""

    session_id: str
    user_id: str
    destination: str = ""
    num_days: int = 0
    query: str = Field(description="用户原始查询")
    itinerary_preview: str = Field(
        default="", description="行程摘要（截取前 500 字符）"
    )
    attraction_names: List[str] = Field(
        default_factory=list, description="本次行程涉及的主要景点"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_document_text(self) -> str:
        """转为可向量化的文档文本（用于 ChromaDB 语义检索）。"""
        return (
            f"目的地: {self.destination}\n"
            f"天数: {self.num_days}\n"
            f"查询: {self.query}\n"
            f"景点: {', '.join(self.attraction_names)}\n"
            f"摘要: {self.itinerary_preview}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 组装后的上下文
# ═══════════════════════════════════════════════════════════════════════════════

class AssembledContext(BaseModel):
    """由 ContextManager 组装后交给 Agent 使用的上下文。"""

    system_context: str = Field(
        default="", description="长时记忆 + 用户画像 组成的系统级上下文"
    )
    recent_messages: List[Dict[str, str]] = Field(
        default_factory=list, description="短时记忆中的最近 N 条消息"
    )
    relevant_memories: List[Dict[str, Any]] = Field(
        default_factory=list, description="语义检索到的相关历史记忆"
    )
    token_count: int = Field(default=0, description="上下文总 token 估算")
    source_labels: List[str] = Field(
        default_factory=list,
        description="上下文来源标签 (如 'user_profile', 'recent_3_messages', 'similar_trip')",
    )

    def to_prompt_fragment(self) -> str:
        """将组装后的上下文转为可直接拼入 prompt 的文本片段。"""
        fragments = []

        if self.system_context:
            fragments.append(self.system_context)

        if self.relevant_memories:
            fragments.append("\n[相关历史记忆]")
            for mem in self.relevant_memories:
                fragments.append(f"- {mem.get('content', '')}")

        if self.recent_messages:
            fragments.append("\n[最近对话]")
            for msg in self.recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                fragments.append(f"{role}: {content}")

        return "\n".join(fragments)


# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermMemoryConfig(BaseModel):
    """长时记忆配置。"""

    db_path: str = "memory.db"
    """SQLite 数据库文件路径"""

    chroma_persist_dir: str = "chroma_memory_db"
    """ChromaDB 向量存储持久化目录"""

    embedding_model: str = "text-embedding-v3"
    """ChromaDB 使用的 embedding 模型（DashScope TextEmbedding 模型名）"""

    max_sessions_to_retrieve: int = 3
    """语义检索时最多返回的历史会话数"""

    similarity_threshold: float = 0.35
    """语义检索的最低相似度阈值"""

# src/memory/long_term.py
"""
长时记忆 (Long-Term Memory)
=============================
跨会话的持久化记忆系统，双引擎架构:

1. **SQLite 引擎** — 结构化数据存储
   - 用户画像 (user_profiles)
   - 会话摘要 (session_summaries)
   - 偏好历史 (preference_history)

2. **ChromaDB 引擎** — 向量语义检索
   - 会话摘要向量化
   - 基于相似度的记忆召回
   - 支持 "类似目的地"、"相似偏好" 等语义查询

设计原则:
    - 双引擎互补: SQLite 做精确查询，ChromaDB 做模糊召回
    - 渐进增强: ChromaDB 不可用时自动降级为纯 SQLite 文本匹配
    - 轻量写入: 写入路径只走 SQLite，ChromaDB 异步/延迟同步
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.memory.embeddings import DashScopeEmbeddingFunction
from src.memory.types import (
    LongTermMemoryConfig,
    SessionSummary,
    UserProfile,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# SQL 建表语句
# ═══════════════════════════════════════════════════════════════════════════════

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id        TEXT PRIMARY KEY,
    preferences    TEXT NOT NULL DEFAULT '{}',   -- JSON
    past_destinations TEXT NOT NULL DEFAULT '[]', -- JSON array
    past_trip_count INTEGER NOT NULL DEFAULT 0,
    favorite_attraction_types TEXT NOT NULL DEFAULT '[]', -- JSON array
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL UNIQUE,
    user_id         TEXT NOT NULL,
    destination     TEXT NOT NULL DEFAULT '',
    num_days        INTEGER NOT NULL DEFAULT 0,
    query           TEXT NOT NULL DEFAULT '',
    itinerary_preview TEXT NOT NULL DEFAULT '',
    attraction_names TEXT NOT NULL DEFAULT '[]',  -- JSON array
    timestamp       TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}'    -- JSON
);

CREATE INDEX IF NOT EXISTS idx_session_user
    ON session_summaries(user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_session_destination
    ON session_summaries(destination);

CREATE TABLE IF NOT EXISTS preference_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT NOT NULL,
    key       TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pref_history_user
    ON preference_history(user_id, timestamp DESC);
"""


class LongTermMemory:
    """跨会话长时记忆。

    使用示例:
        ltm = LongTermMemory()
        await ltm.initialize()

        # 获取用户画像
        profile = await ltm.get_user_profile("user_123")

        # 保存会话摘要
        await ltm.save_session_summary(summary)

        # 语义检索相似历史
        memories = await ltm.retrieve_similar("北京三日游", user_id="user_123")

        # 更新偏好
        await ltm.update_preferences("user_123", {"interests": ["美食", "历史"]})
    """

    # ── 构造函数 ──────────────────────────────────────────────────────────

    def __init__(self, config: Optional[LongTermMemoryConfig] = None):
        self.config = config or LongTermMemoryConfig()
        self._db_path = Path(self.config.db_path)
        self._conn: Optional[sqlite3.Connection] = None

        # ChromaDB 客户端（延迟初始化）
        self._chroma_client = None
        self._chroma_collection = None
        self._chroma_available = False

    # ── 初始化 ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """初始化双引擎存储。"""
        # SQLite
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(CREATE_TABLES_SQL)
        self._conn.commit()
        logger.info(f"SQLite 长时记忆已就绪: {self._db_path}")

        # ChromaDB
        await self._init_chroma()

    async def _init_chroma(self) -> None:
        """尝试初始化 ChromaDB 向量存储。"""
        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = str(Path(self.config.chroma_persist_dir).absolute())
            self._chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

            # 创建 DashScope embedding 函数
            embedding_fn = DashScopeEmbeddingFunction(
                model=self.config.embedding_model,
                text_type="document",
            )

            collection_name = "trip_memories"

            # 获取或创建 collection
            try:
                self._chroma_collection = self._chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_fn,
                )
            except Exception:
                # 可能不存在，或 embedding function 不兼容（旧 collection 用的其他模型）
                try:
                    self._chroma_client.delete_collection(name=collection_name)
                    logger.info(
                        "已删除旧的 ChromaDB collection (embedding 模型不兼容)，将重新创建"
                    )
                except Exception:
                    pass
                self._chroma_collection = self._chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "旅行规划长时记忆"},
                    embedding_function=embedding_fn,
                )

            self._chroma_available = True
            count = self._chroma_collection.count()
            logger.info(
                f"ChromaDB 向量存储已就绪: {persist_dir} "
                f"(collection=trip_memories, docs={count}, "
                f"model={self.config.embedding_model})"
            )

        except ImportError:
            logger.warning("chromadb 未安装，向量检索不可用，降级为纯 SQLite 模式")
        except Exception as e:
            logger.warning(f"ChromaDB 初始化失败: {e}，降级为纯 SQLite 模式")

    # ── 用户画像 CRUD ────────────────────────────────────────────────────

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """获取或创建用户画像。

        如果用户不存在，自动创建空画像。
        """
        if not self._conn:
            raise RuntimeError("LongTermMemory 未初始化，请先调用 initialize()")

        row = self._conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()

        if row is None:
            profile = UserProfile(user_id=user_id)
            self._conn.execute(
                """INSERT INTO user_profiles
                   (user_id, preferences, past_destinations, past_trip_count,
                    favorite_attraction_types, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile.user_id,
                    json.dumps(profile.preferences, ensure_ascii=False),
                    json.dumps(profile.past_destinations, ensure_ascii=False),
                    profile.past_trip_count,
                    json.dumps(profile.favorite_attraction_types, ensure_ascii=False),
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
            return profile

        return UserProfile(
            user_id=row["user_id"],
            preferences=json.loads(row["preferences"]),
            past_destinations=json.loads(row["past_destinations"]),
            past_trip_count=row["past_trip_count"],
            favorite_attraction_types=json.loads(row["favorite_attraction_types"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def update_preferences(
        self, user_id: str, new_prefs: Dict[str, Any]
    ) -> UserProfile:
        """更新用户偏好（合并模式）。"""
        if not self._conn:
            raise RuntimeError("LongTermMemory 未初始化")

        profile = await self.get_user_profile(user_id)

        # 记录变更历史
        for key, value in new_prefs.items():
            old_value = profile.preferences.get(key)
            if old_value != value:
                self._conn.execute(
                    """INSERT INTO preference_history
                       (user_id, key, old_value, new_value, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        key,
                        json.dumps(old_value, ensure_ascii=False) if old_value is not None else None,
                        json.dumps(value, ensure_ascii=False) if value is not None else None,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

        # 合并偏好
        profile.merge_preferences(new_prefs)

        # 写回数据库
        self._conn.execute(
            """UPDATE user_profiles
               SET preferences = ?,
                   past_destinations = ?,
                   past_trip_count = ?,
                   favorite_attraction_types = ?,
                   updated_at = ?
               WHERE user_id = ?""",
            (
                json.dumps(profile.preferences, ensure_ascii=False),
                json.dumps(profile.past_destinations, ensure_ascii=False),
                profile.past_trip_count,
                json.dumps(profile.favorite_attraction_types, ensure_ascii=False),
                profile.updated_at.isoformat(),
                user_id,
            ),
        )
        self._conn.commit()
        logger.debug(f"用户偏好已更新: user={user_id} keys={list(new_prefs.keys())}")
        return profile

    # ── 会话摘要 ──────────────────────────────────────────────────────────

    async def save_session_summary(self, summary: SessionSummary) -> None:
        """保存会话摘要（同时写入 SQLite 和 ChromaDB）。

        在会话成功完成后调用。
        """
        if not self._conn:
            raise RuntimeError("LongTermMemory 未初始化")

        # ── SQLite 写入 ───────────────────────────────────────────────
        self._conn.execute(
            """INSERT OR REPLACE INTO session_summaries
               (session_id, user_id, destination, num_days, query,
                itinerary_preview, attraction_names, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                summary.session_id,
                summary.user_id,
                summary.destination,
                summary.num_days,
                summary.query,
                summary.itinerary_preview[:500],
                json.dumps(summary.attraction_names, ensure_ascii=False),
                summary.timestamp.isoformat(),
                json.dumps(summary.metadata, ensure_ascii=False),
            ),
        )

        # 更新用户画像的去过目的地和行程计数
        profile = await self.get_user_profile(summary.user_id)
        if summary.destination and summary.destination not in profile.past_destinations:
            profile.past_destinations.append(summary.destination)
        profile.past_trip_count += 1

        self._conn.execute(
            """UPDATE user_profiles
               SET past_destinations = ?, past_trip_count = ?, updated_at = ?
               WHERE user_id = ?""",
            (
                json.dumps(profile.past_destinations, ensure_ascii=False),
                profile.past_trip_count,
                datetime.now(timezone.utc).isoformat(),
                summary.user_id,
            ),
        )
        self._conn.commit()

        # ── ChromaDB 写入 ─────────────────────────────────────────────
        if self._chroma_available and self._chroma_collection:
            try:
                doc_text = summary.to_document_text()
                self._chroma_collection.add(
                    documents=[doc_text],
                    metadatas=[
                        {
                            "session_id": summary.session_id,
                            "user_id": summary.user_id,
                            "destination": summary.destination,
                            "num_days": summary.num_days,
                            "timestamp": summary.timestamp.isoformat(),
                        }
                    ],
                    ids=[summary.session_id],
                )
                logger.debug(f"ChromaDB 索引已更新: session={summary.session_id}")
            except Exception as e:
                logger.warning(f"ChromaDB 写入失败: {e}")

        logger.info(
            f"会话已归档: session={summary.session_id} "
            f"dest={summary.destination} user={summary.user_id}"
        )

    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[SessionSummary]:
        """获取用户的历史会话摘要（按时间倒序）。"""
        if not self._conn:
            return []

        rows = self._conn.execute(
            """SELECT * FROM session_summaries
               WHERE user_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()

        return [
            SessionSummary(
                session_id=row["session_id"],
                user_id=row["user_id"],
                destination=row["destination"],
                num_days=row["num_days"],
                query=row["query"],
                itinerary_preview=row["itinerary_preview"],
                attraction_names=json.loads(row["attraction_names"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    # ── 语义检索 ──────────────────────────────────────────────────────────

    async def retrieve_similar(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """语义检索与查询相似的历史记忆。

        Args:
            query: 查询文本（如 "北京三日游"）
            user_id: 限定用户（None = 全局检索）
            limit: 返回数量上限

        Returns:
            [{"content": "...", "score": 0.85, "metadata": {...}}, ...]
        """
        max_results = limit or self.config.max_sessions_to_retrieve

        if self._chroma_available and self._chroma_collection:
            return await self._chroma_retrieve(query, user_id, max_results)
        else:
            return await self._sqlite_fallback_retrieve(query, user_id, max_results)

    async def _chroma_retrieve(
        self, query: str, user_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """ChromaDB 语义检索。"""
        try:
            where_filter = None
            if user_id:
                where_filter = {"user_id": user_id}

            results = self._chroma_collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter,
            )

            memories = []
            if results and results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = (
                        results["metadatas"][0][i]
                        if results.get("metadatas") and results["metadatas"][0]
                        else {}
                    )
                    distance = (
                        results["distances"][0][i]
                        if results.get("distances") and results["distances"][0]
                        else 1.0
                    )
                    # ChromaDB 默认返回距离（越小越相似），转为相似度
                    similarity = max(0.0, 1.0 - distance)

                    if similarity >= self.config.similarity_threshold:
                        memories.append(
                            {
                                "content": doc,
                                "score": round(similarity, 4),
                                "metadata": meta,
                            }
                        )

            logger.debug(
                f"ChromaDB 检索: query='{query[:40]}...' → {len(memories)} 条"
            )
            return memories

        except Exception as e:
            logger.warning(f"ChromaDB 检索失败: {e}，降级为 SQLite")
            return await self._sqlite_fallback_retrieve(query, user_id, limit)

    async def _sqlite_fallback_retrieve(
        self, query: str, user_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """纯 SQLite 文本匹配回退（无向量检索时使用）。

        使用简单的关键词匹配 + 时间衰减排序。
        """
        if not self._conn:
            return []

        # 提取查询关键词
        keywords = self._extract_keywords(query)

        sql = """SELECT * FROM session_summaries"""
        params: List[Any] = []

        conditions = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit * 3)  # 多取一些用于关键词过滤

        rows = self._conn.execute(sql, params).fetchall()

        memories = []
        for row in rows:
            summary = SessionSummary(
                session_id=row["session_id"],
                user_id=row["user_id"],
                destination=row["destination"],
                num_days=row["num_days"],
                query=row["query"],
                itinerary_preview=row["itinerary_preview"],
                attraction_names=json.loads(row["attraction_names"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]),
            )

            # 关键词匹配打分
            doc_text = summary.to_document_text()
            score = self._keyword_match_score(keywords, doc_text)

            if score > 0:
                # 时间衰减: 越旧的记忆分越低
                days_ago = (
                    datetime.now(timezone.utc) - summary.timestamp
                ).days
                time_decay = max(0.3, 1.0 - days_ago / 365.0)
                final_score = score * time_decay

                if final_score >= self.config.similarity_threshold:
                    memories.append(
                        {
                            "content": summary.itinerary_preview or summary.query,
                            "score": round(final_score, 4),
                            "metadata": {
                                "session_id": summary.session_id,
                                "user_id": summary.user_id,
                                "destination": summary.destination,
                                "num_days": summary.num_days,
                                "timestamp": summary.timestamp.isoformat(),
                            },
                        }
                    )

        # 按分数排序取 top-N
        memories.sort(key=lambda x: x["score"], reverse=True)
        memories = memories[:limit]

        logger.debug(
            f"SQLite 关键词检索: query='{query[:40]}...' → {len(memories)} 条"
        )
        return memories

    # ── 工具方法 ──────────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> List[str]:
        """简单的中文关键词提取（基于字符级分词）。

        生产环境可替换为 jieba 或其他分词器。
        """
        # 提取 2-4 字的片段作为关键词
        keywords = []
        for length in [2, 3, 4]:
            for i in range(len(text) - length + 1):
                seg = text[i : i + length]
                # 过滤纯符号/数字
                if any("一" <= c <= "鿿" for c in seg):
                    keywords.append(seg)
        # 去重并保留有意义的片段
        return list(dict.fromkeys(keywords))[-30:]

    def _keyword_match_score(self, keywords: List[str], text: str) -> float:
        """关键词匹配打分。"""
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw in text)
        return hits / len(keywords)

    # ── 景点特征存储 ──────────────────────────────────────────────────────

    async def save_attraction_features(
        self,
        attractions: List[Dict[str, Any]],
        user_id: str,
        destination: str,
    ) -> None:
        """将景点特征作为独立向量文档存储，供未来语义检索。

        每个景点生成一条文档：「{destination} | {name} | {reason}」，
        存储后可通过 retrieve_similar("古城 历史") 召回匹配的景点偏好。

        Args:
            attractions: 景点列表（至少包含 name, reason 字段）
            user_id: 用户标识
            destination: 目的地城市
        """
        if not self._chroma_available or not self._chroma_collection:
            return

        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for attr in attractions:
            name = attr.get("name", "")
            reason = attr.get("reason", "")
            if not name:
                continue
            doc_text = f"目的地: {destination} | 景点: {name} | 特征: {reason}"
            docs.append(doc_text)
            metadatas.append({
                "user_id": user_id,
                "destination": destination,
                "attraction_name": name,
                "type": "attraction_feature",
            })
            ids.append(f"attr_{user_id}_{destination}_{name}")

        if docs:
            try:
                self._chroma_collection.upsert(
                    documents=docs,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.debug(f"景点特征已写入向量库: {len(docs)}条")
            except Exception as e:
                logger.warning(f"景点特征写入 ChromaDB 失败: {e}")

    # ── 清理 ──────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """关闭所有连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._chroma_client = None
        self._chroma_collection = None
        self._chroma_available = False
        logger.info("LongTermMemory 已关闭")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()

# src/memory/embeddings.py
"""
自定义 ChromaDB EmbeddingFunction 实现。

使用示例:
    from src.memory.embeddings import DashScopeEmbeddingFunction

    ef = DashScopeEmbeddingFunction(
        model="text-embedding-v3",
        api_key="sk-xxx",          # 可选，默认读 DASHSCOPE_API_KEY 环境变量
        text_type="document",       # "document" | "query"
    )
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class DashScopeEmbeddingFunction(EmbeddingFunction):
    """
    参数:
        model: 模型名称，默认 "text-embedding-v3"
        text_type: "document" (文档入库) 或 "query" (查询)，默认 "document"
    """

    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: Optional[str] = None,
        text_type: str = "document",
    ):
        self._model = model
        self._api_key = api_key
        self._text_type = text_type

    def __call__(self, input: Documents) -> Embeddings:
        """对一批文本进行 embedding。

        Args:
            input: 文本列表

        Returns:
            向量列表，每个向量是 float 列表
        """
        if not input:
            return []

        import dashscope
        from dashscope import TextEmbedding

        # 设置 API Key（优先使用传入的 key，其次环境变量）
        api_key = self._api_key or os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            dashscope.api_key = api_key

        resp = TextEmbedding.call(
            model=self._model,
            input=list(input),
            text_type=self._text_type,
        )

        if resp.status_code != 200:
            error_msg = (
                f"DashScope embedding 请求失败: "
                f"status={resp.status_code} code={resp.code} "
                f"message={resp.message}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 提取 embedding 向量
        embeddings: Embeddings = []
        for emb_item in resp.output.get("embeddings", []):
            embeddings.append(emb_item["embedding"])
        return embeddings

    @property
    def name(self) -> str:
        """ChromaDB 要求的 name 属性，用于序列化/反序列化。"""
        return f"dashscope:{self._model}:{self._text_type}"

    def __repr__(self) -> str:
        return (
            f"DashScopeEmbeddingFunction("
            f"model={self._model!r}, "
            f"text_type={self._text_type!r})"
        )

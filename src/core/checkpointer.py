"""检查点管理 - 支持SQLite和PostgreSQL，用于持久化LangGraph状态"""

import os
from typing import Optional, Union
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
import logging

logger = logging.getLogger(__name__)


def create_checkpointer(
    db_url: Optional[str] = None,
    use_postgres: bool = False,
    sqlite_path: str = "checkpoints.db"
) -> Union[SqliteSaver, PostgresSaver]:
    """
    创建检查点实例
    
    Args:
        db_url: 数据库连接URL（对于PostgreSQL必须提供）
        use_postgres: 是否使用PostgreSQL，默认False使用SQLite
        sqlite_path: SQLite文件路径（use_postgres=False时生效）
    
    Returns:
        SqliteSaver 或 PostgresSaver 实例
    """
    if use_postgres:
        if not db_url:
            raise ValueError("使用PostgreSQL时必须提供 db_url")
        logger.info("使用PostgreSQL检查点: %s", db_url)
        # PostgreSQL连接需要异步上下文，这里返回同步包装器
        # 注意：PostgresSaver需要使用async with，但我们可以返回同步版本
        # 实际LangGraph提供了 PostgresSaver.from_conn_string
        try:
            # 同步版本的PostgresSaver (langgraph >= 0.2.0)
            saver = PostgresSaver.from_conn_string(db_url)
            # 需要调用setup()创建表，但通常由用户手动调用
            # 这里我们返回saver，由调用者决定是否setup
            logger.info("PostgreSQL检查点初始化成功")
            return saver
        except Exception as e:
            logger.error("PostgreSQL检查点初始化失败: %s", e)
            raise
    else:
        logger.info("使用SQLite检查点: %s", sqlite_path)
        # 确保目录存在
        dir_path = os.path.dirname(sqlite_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        try:
            saver = SqliteSaver.from_conn_string(sqlite_path)
            logger.info("SQLite检查点初始化成功")
            return saver
        except Exception as e:
            logger.error("SQLite检查点初始化失败: %s", e)
            raise


def setup_postgres_checkpoint(saver: PostgresSaver) -> None:
    """
    为PostgreSQL检查点创建必要的表（需要异步调用）
    实际使用时，应该在应用启动时调用 saver.setup()
    但同步版本无法直接调用，需要async context manager
    """
    # 由于PostgresSaver需要async，这里仅提供同步包装方法
    # 建议在async启动时直接调用 await saver.setup()
    pass
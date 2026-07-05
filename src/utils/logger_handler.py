"""日志处理模块 (Logger Handler)

提供统一的日志管理功能：
- 同时输出到控制台和按天分割的日志文件
- logs/ 目录自动创建
- 三类日志文件：app（全量）、error（仅错误）、access（API 访问）
- 每个 logger 独立配置级别，互不干扰
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# ── 项目根目录 & 日志目录 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── 日志格式 ──────────────────────────────────────────────────────
DETAIL_FORMAT = logging.Formatter(
    "%(asctime)s | %(name)-20s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SIMPLE_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ACCESS_FORMAT = logging.Formatter(
    "%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── 今天的日期字符串（用于文件名）──────────────────────────────────
_today = datetime.now().strftime("%Y%m%d")

# ── 全局状态：防止重复初始化 ──────────────────────────────────────
_initialized = False


def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """一次性初始化整个应用的日志系统。

    创建三组日志文件（按天命名）：
    - logs/app_YYYYMMDD.log    : 全量应用日志
    - logs/error_YYYYMMDD.log  : 仅 ERROR 及以上
    - logs/access_YYYYMMDD.log : API 请求/响应日志
    """
    global _initialized
    if _initialized:
        return

    # ── 根 logger：捕获所有未处理的日志 ───────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 控制台 handler（INFO 以上才显示）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(SIMPLE_FORMAT)
    root.addHandler(console)

    # ── 应用全量日志（DEBUG 以上全写）──────────────────────────────
    app_handler = logging.FileHandler(
        LOG_DIR / f"app_{_today}.log", encoding="utf-8"
    )
    app_handler.setLevel(file_level)
    app_handler.setFormatter(DETAIL_FORMAT)
    root.addHandler(app_handler)

    # ── 错误日志（仅 ERROR + CRITICAL）─────────────────────────────
    error_handler = logging.FileHandler(
        LOG_DIR / f"error_{_today}.log", encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(DETAIL_FORMAT)
    root.addHandler(error_handler)

    # ── 抑制第三方库的日志噪音 ─────────────────────────────────
    for noisy in ("httpx", "httpcore", "urllib3", "aiosqlite", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 只显示错误（抑制 reload 文件变更和 langsmith 403 警告）
    logging.getLogger("watchfiles").setLevel(logging.ERROR)
    logging.getLogger("langsmith").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger。

    使用方式：
        from src.utils.logger_handler import get_logger
        logger = get_logger(__name__)
        logger.info("something happened")
    """
    return logging.getLogger(name)


def get_access_logger() -> logging.Logger:
    """获取 API 访问日志专用 logger。

    写入 logs/access_YYYYMMDD.log，仅文件输出，不污染控制台。
    """
    access = logging.getLogger("access")
    if not access.handlers:
        access.setLevel(logging.INFO)
        access.propagate = False  # 不重复输出到 root handler
        handler = logging.FileHandler(
            LOG_DIR / f"access_{_today}.log", encoding="utf-8"
        )
        handler.setFormatter(ACCESS_FORMAT)
        access.addHandler(handler)
    return access

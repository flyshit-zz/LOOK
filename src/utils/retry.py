# src/utils/retry.py
"""异步重试工具 — 指数退避 + 日志记录，无需额外依赖"""
import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar, Awaitable, Tuple, Type

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def with_retry(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = "",
    **kwargs,
) -> T:
    """对异步函数执行指数退避重试。

    Args:
        func: 要重试的异步函数
        max_retries: 最大重试次数（含首次，默认 3）
        base_delay: 基础延迟秒数，每次翻倍
        retryable_exceptions: 可重试的异常类型元组
        label: 日志中显示的操作名称
        *args, **kwargs: 传递给 func 的参数

    Returns:
        func 的返回值

    Raises:
        最后一次失败时的异常（重试耗尽后）
    """
    label = label or getattr(func, "__name__", "operation")
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"[{label}] 第 {attempt}/{max_retries} 次失败: {e}. "
                    f"{delay:.1f}s 后重试..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"[{label}] 重试 {max_retries} 次全部失败: {e}")
                raise last_error

    # 理论上不会到这里
    raise last_error  # type: ignore[misc]


def retryable(
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = "",
):
    """装饰器版本的异步重试。

    Usage:
        @retryable(max_retries=3, base_delay=1.0)
        async def my_func(): ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_retry(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                retryable_exceptions=retryable_exceptions,
                label=label or func.__name__,
                **kwargs,
            )
        return wrapper
    return decorator

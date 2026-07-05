# src/core/graph.py
import os
import aiosqlite
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.core.state import TravelState
from src.core.routing import router
from src.agents.supervisor import SupervisorAgent
from src.agents.attraction import AttractionAgent
from src.agents.stay_and_dine import StayAndDineAgent
from src.agents.itinerary import ItineraryAgent


async def build_graph(mcp_adapter, memory_store=None):
    """构建 LangGraph 工作流。

    Args:
        mcp_adapter: MCP 工具适配器
        memory_store: 可选的 MemoryStore 实例，为 Agent 提供记忆能力
    """
    # ── LangSmith 追踪回调 ─────────────────────────────────────────
    tracing_enabled = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    callbacks = []
    if tracing_enabled:
        try:
            from langchain_core.tracers.langchain import LangChainTracer
            tracer = LangChainTracer(
                project_name=os.environ.get("LANGSMITH_PROJECT", "travel-assistant-mvp"),
            )
            callbacks.append(tracer)
        except Exception:
            pass  # LangSmith 不可用时静默降级

    # 初始化Agent（注入 memory_store）
    supervisor = SupervisorAgent(memory_store=memory_store)
    attraction = AttractionAgent(mcp_adapter, memory_store=memory_store)
    stay_and_dine = StayAndDineAgent(mcp_adapter, memory_store=memory_store)
    itinerary = ItineraryAgent(memory_store=memory_store)

    # 构建图
    workflow = StateGraph(TravelState)

    workflow.add_node("supervisor", supervisor.execute)
    workflow.add_node("attraction", attraction.execute)
    workflow.add_node("stay_and_dine", stay_and_dine.execute)
    workflow.add_node("itinerary", itinerary.execute)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "attraction": "attraction",
            "stay_and_dine": "stay_and_dine",
            "itinerary": "itinerary",
            "end": END
        }
    )
    workflow.add_edge("attraction", "supervisor")
    workflow.add_edge("stay_and_dine", "supervisor")
    workflow.add_edge("itinerary", "supervisor")

    # 使用SQLite检查点
    conn = await aiosqlite.connect("checkpoints.db")
    checkpointer = AsyncSqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)
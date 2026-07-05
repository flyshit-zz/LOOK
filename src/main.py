# src/main.py
import sys
import os
import time
import traceback
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# ── 最早：加载环境变量 ────────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv(raise_error_if_not_found=False)
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()

# ── 初始化日志系统 ────────────────────────────────────────────────
from src.utils.logger_handler import setup_logging, get_logger, get_access_logger
setup_logging()

logger = get_logger(__name__)
access_logger = get_access_logger()

import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langsmith import Client as LangSmithClient
from langsmith import traceable
from src.trip_mcp.registry import MCPServerRegistry
from src.trip_mcp.client import MCPClient
from src.trip_mcp.adapter import MCPAdapter
from src.core.graph import build_graph
from src.memory import MemoryStore

app = FastAPI(title="旅行助手MVP")

# ── CORS 中间件 ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求日志中间件 ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个 HTTP 请求的方法、路径、状态码和耗时"""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    access_logger.info(
        f"{request.method:6s} {request.url.path:20s} → {response.status_code} "
        f"({duration_ms:.0f}ms)"
    )

    # 慢请求警告（超过 5 秒）
    if duration_ms > 5000:
        logger.warning(
            f"慢请求: {request.method} {request.url.path} 耗时 {duration_ms:.0f}ms"
        )

    return response


# ── 数据模型 ──────────────────────────────────────────────────────
class PlanRequest(BaseModel):
    query: str
    session_id: str = "default"


# ── 全局实例 ──────────────────────────────────────────────────────
graph = None
memory_store: MemoryStore = None
langsmith_client: LangSmithClient = None
langsmith_available: bool = False


# ── 启动事件 ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global graph, memory_store, langsmith_client, langsmith_available
    logger.info("=" * 60)
    logger.info("旅行助手服务启动中...")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"日志目录: {PROJECT_ROOT / 'logs'}")
    logger.info(f"DeepSeek API Key: {'已设置' if os.environ.get('DEEPSEEK_API_KEY') else '❌ 未设置'}")
    logger.info(f"高德 API Key: {'已设置' if os.environ.get('AMAP_API_KEY') else '❌ 未设置'}")

    # ── 初始化 LangSmith ───────────────────────────────────────────
    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
    langsmith_tracing = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    langsmith_project = os.environ.get("LANGSMITH_PROJECT", "default")

    if langsmith_tracing and langsmith_api_key:
        try:
            langsmith_client = LangSmithClient(
                api_key=langsmith_api_key,
            )
            # 验证连接：尝试列出项目
            langsmith_client.list_projects()
            langsmith_available = True
            logger.info(f"✅ LangSmith 追踪已启用 (项目: {langsmith_project})")
        except Exception as e:
            logger.warning(f"⚠️  LangSmith 连接失败: {e}")
            logger.warning("   LangSmith 追踪将不可用，但服务正常运行")
            langsmith_available = False
    else:
        if not langsmith_tracing:
            logger.info("ℹ️  LangSmith 追踪未启用 (LANGSMITH_TRACING=false)")
        elif not langsmith_api_key:
            logger.warning("⚠️  LANGSMITH_API_KEY 未设置，LangSmith 追踪不可用")

    try:
        # ── 初始化记忆模块 ──────────────────────────────────────────
        memory_store = MemoryStore()
        await memory_store.initialize()
        logger.info("记忆模块已就绪")

        config = MCPServerRegistry().load_config()
        logger.info(f"MCP 服务器: {list(config.keys())}")
        client = await MCPClient(config).initialize()
        adapter = await MCPAdapter(client).initialize()
        logger.info(f"MCP 适配器就绪，工具数: {len(adapter.get_tools())}")
        graph = await build_graph(adapter, memory_store=memory_store)
        logger.info("Graph 构建完成，服务就绪 ")
    except Exception as e:
        logger.critical(f"启动失败: {e}")
        logger.critical(traceback.format_exc())
        raise

    logger.info("=" * 60)


# ── API 路由 ──────────────────────────────────────────────────────
@traceable(
    run_type="chain",
    name="travel-planning",
    project_name=os.environ.get("LANGSMITH_PROJECT", "travel-assistant-mvp"),
)
async def _execute_plan(query: str, session_id: str, user_id: str, memory_store) -> dict:
    """执行旅行规划的核心逻辑（被 @traceable 装饰，自动上报 LangSmith）。"""
    initial_state = {
        "user_input": query,
        "session_id": session_id,
        "user_id": user_id,
        "messages": [],
        "interests": [],
        "is_complete": False,
    }

    # 每次请求用唯一 thread_id，避免从旧 checkpoint 恢复已完成状态
    import uuid
    config = {"configurable": {"thread_id": f"{session_id}_{uuid.uuid4().hex[:8]}"}}
    result = await graph.ainvoke(initial_state, config=config)
    return result


@app.post("/plan")
async def plan(request: PlanRequest):
    """旅行规划接口"""
    if graph is None:
        logger.error("收到请求但 graph 未初始化")
        raise HTTPException(503, "系统尚未初始化完成，请等待片刻后重试")

    logger.info(f"收到规划请求: session={request.session_id} query={request.query[:60]}...")

    # ── 记忆模块: 创建/获取会话并记录用户输入 ──────────────────────────
    user_id = "default"  # TODO: 从认证层获取真实 user_id
    if memory_store is not None:
        memory_store.create_session(
            session_id=request.session_id,
            user_id=user_id,
            initial_user_input=request.query,
        )

    try:
        result = await _execute_plan(
            query=request.query,
            session_id=request.session_id,
            user_id=user_id,
            memory_store=memory_store,
        )

        destination = result.get("destination", "")
        next_agent = result.get("next_agent", "")
        is_complete = result.get("is_complete", False)
        itinerary_len = len(result.get("itinerary", ""))

        logger.info(
            f"✅ 请求完成: dest={destination} agent={next_agent} "
            f"complete={is_complete} itinerary_len={itinerary_len}"
        )

        # ── 记忆模块: 行程完成时归档会话 ──────────────────────────────
        if is_complete and memory_store is not None:
            try:
                await memory_store.archive_session(
                    session_id=request.session_id,
                    state=result,
                    user_id=user_id,
                )
            except Exception as e:
                logger.warning(f"会话归档失败: {e}")

        # ── 提取景点数据（含 POI ID，供前端图片预览用） ──────────────
        attractions_data = result.get("attractions", [])
        daily_routes_data = result.get("daily_routes", [])

        return {
            "session_id": request.session_id,
            "itinerary": result.get("itinerary", "生成中..."),
            "is_complete": is_complete,
            "destination": destination,
            "attractions": [
                {
                    "id": a.get("id", ""),
                    "name": a.get("name", ""),
                    "address": a.get("address", ""),
                    "lat": a.get("lat", 0.0),
                    "lng": a.get("lng", 0.0),
                    "rating": a.get("rating", 0.0),
                }
                for a in attractions_data
            ],
        }

    except Exception as e:
        logger.error(f"❌ 处理请求失败: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"处理请求失败: {str(e)}")


@app.get("/api/poi/photos")
async def get_poi_photos(
    name: str = Query(..., description="景点名称"),
    city: str = Query("", description="所在城市（可选）"),
):
    """获取景点图片 —— 通过高德 POI 搜索 API（extensions=all 含图片）。

    返回:
        photos: 图片列表 [{url, title, provider}]
        static_map_url: 静态地图 URL（兜底）
        poi_name, poi_address: 匹配到的 POI 信息
    """
    amap_key = os.environ.get("AMAP_API_KEY", "")
    if not amap_key:
        raise HTTPException(500, "AMAP_API_KEY 未配置")

    # ── 调用高德 POI 文本搜索（extensions=all 返回图片） ──────────
    params: dict = {
        "key": amap_key,
        "keywords": name,
        "extensions": "all",
        "offset": 5,
    }
    if city:
        params["city"] = city

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                "https://restapi.amap.com/v3/place/text",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error(f"高德 API 请求失败: {e}")
            raise HTTPException(502, f"高德 API 请求失败: {e}")

    if data.get("status") != "1":
        logger.warning(f"高德 API 返回异常: {data}")
        raise HTTPException(502, f"高德 API 返回异常: {data.get('info', '未知错误')}")

    pois = data.get("pois", [])
    if not pois:
        return {
            "poi_name": name,
            "poi_address": "",
            "photos": [],
            "static_map_url": None,
            "message": "未找到匹配的 POI",
        }

    # ── 取最佳匹配（按名称精确匹配优先，否则取第一个） ────────────
    best = pois[0]
    for p in pois:
        if p.get("name", "") == name:
            best = p
            break

    poi_name = best.get("name", name)
    poi_address = best.get("address", "")
    location = best.get("location", "")  # "lng,lat"
    photos = best.get("photos", []) or []

    # ── 生成静态地图 URL（无论有无照片都做兜底） ──────────────────
    static_map_url = None
    if location:
        static_map_url = (
            f"https://restapi.amap.com/v3/staticmap"
            f"?location={location}&zoom=15&size=600*400"
            f"&markers=mid,,A:{location}"
            f"&key={amap_key}"
        )

    logger.info(
        f"POI 图片查询: name={name} city={city} → "
        f"matched={poi_name} photos={len(photos)}"
    )

    return {
        "poi_name": poi_name,
        "poi_address": poi_address,
        "photos": [
            {
                "url": p.get("url", ""),
                "title": p.get("title", ""),
                "provider": p.get("provider", ""),
            }
            for p in photos
        ],
        "static_map_url": static_map_url,
    }


@app.get("/")
async def serve_frontend():
    """托管前端页面"""
    index_path = PROJECT_ROOT / "index.html"
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "graph_ready": graph is not None,
        "memory_active": memory_store is not None,
        "memory_sessions": memory_store.active_sessions if memory_store else 0,
        "langsmith": {
            "enabled": os.environ.get("LANGSMITH_TRACING", "").lower() == "true",
            "available": langsmith_available,
            "project": os.environ.get("LANGSMITH_PROJECT", "default"),
        },
    }


@app.on_event("shutdown")
async def shutdown():
    """应用关闭时清理资源。"""
    global memory_store
    if memory_store is not None:
        logger.info("正在关闭记忆模块...")
        await memory_store.close()
        logger.info("记忆模块已关闭")

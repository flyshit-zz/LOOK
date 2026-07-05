# src/agents/attraction.py
import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.base import BaseAgent
from src.utils.geo import cluster_by_location, optimize_route
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

# ── Attraction 系统提示词 ─────────────────────────────────────────────
ATTRACTION_SYSTEM_PROMPT = """你是一个专业的旅游景点搜索与规划助手。你可以调用高德地图 MCP 工具来搜索和获取景点信息。

## 可用工具说明
你拥有完整的高德地图 MCP 工具集，主要包括：
- **maps_text_search**: 关键字搜索 POI，支持按城市过滤。参数: keywords(关键词), city(城市), citylimit(是否限制同城)
- **maps_search_detail**: 根据 POI ID 获取详细信息，包含坐标、评分、地址等。参数: id(POI ID)
- **maps_geo**: 地理编码，将地址转换为坐标
- **maps_regeocode**: 逆地理编码，将坐标转换为地址
- **maps_direction_***: 各类路线规划工具（步行、驾车、骑行、公交）

## 工作流程
1. **搜索景点**: 使用 maps_text_search，以 "{destination}景点" 或 "{destination}必去景点" 为关键词搜索
2. **获取详情**: 对搜索结果中的每个景点，使用 maps_search_detail 获取坐标和评分
3. **筛选排序**: 剔除低质量景点（评分 < 3.0 或坐标缺失），按评分降序排列
4. **输出结果**: 以 JSON 格式输出最终景点列表

## 搜索策略
- 关键词要多样化："{destination}热门景点"、"{destination}必去"、"{destination}公园" 等
- 优先保留 4A/5A 级景区、地标建筑、热门打卡地
- 兼顾自然景观与人文景观的多样性
- 最终保留 8-15 个优质景点

## 输出格式要求
在完成所有工具调用和分析后，你必须在最终回复中输出一个 JSON 数组，格式如下：
```json
[
  {
    "name": "景点名称",
    "address": "详细地址",
    "rating": 4.5,
    "reason": "推荐理由"
  }
]
```
注意：
- rating 必须是数字类型（float），缺失时填 0.0
- reason 用中文简述为什么推荐这个景点（10-30字）
- 按 rating 从高到低排序
- 只输出 JSON 数组，不要包含其他文字"""


class AttractionAgent(BaseAgent):
    """景点规划 Agent —— 通过 LLM + MCP 工具自主搜索和筛选景点"""

    def __init__(self, mcp_adapter, llm=None, memory_store=None):
        super().__init__("attraction", "景点规划", memory_store=memory_store)
        self.mcp_adapter = mcp_adapter
        self.system_prompt = ATTRACTION_SYSTEM_PROMPT

        # ── 提取所有 MCP 工具 ──────────────────────────────────────
        self.tools = mcp_adapter.get_tools()
        tool_names = [t.name for t in self.tools]
        logger.info(f"AttractionAgent 加载 {len(self.tools)} 个 MCP 工具: {tool_names}")

        # ── 初始化 LLM ─────────────────────────────────────────────
        logger.info("初始化 AttractionAgent LLM（DeepSeek）...")
        self.llm = llm or init_chat_model(
            model="deepseek-chat",
            temperature=0.3,
        )

        # ── 创建内部 LLM Agent（带 MCP 工具） ──────────────────────
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=InMemorySaver(),
        )
        logger.info("AttractionAgent 内部 LLM Agent 初始化完成")

    # ── 主执行入口 ──────────────────────────────────────────────────
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        destination = state.get("destination", "")
        num_days = state.get("num_days", 3)

        logger.info(f"开始景点规划: dest={destination} days={num_days}")

        # ── 获取记忆上下文（用户画像 + 历史兴趣） ──────────────────
        memory_interests: List[str] = []
        memory_attraction_types: List[str] = []
        if self.memory_store is not None:
            memory_ctx = await self.get_memory_context(state)
            if memory_ctx:
                # 从 relevant_memories 中提取历史景点偏好
                relevant = memory_ctx.get("relevant_memories", [])
                for mem in relevant:
                    content = mem.get("content", "")
                    # 从历史记忆文档中提取景点名称作为兴趣参考
                    if "景点:" in content:
                        for part in content.split("|"):
                            part = part.strip()
                            if part.startswith("景点:") and len(part) > 3:
                                name = part[3:].strip()
                                if name and name not in memory_interests:
                                    memory_interests.append(name)
                # 从 user_profile 中提取兴趣标签和偏好景点类型
                sys_ctx = memory_ctx.get("system_context", "")
                if "兴趣:" in sys_ctx:
                    match = re.search(r"兴趣:\s*(.+)", sys_ctx)
                    if match:
                        memory_interests.extend(
                            [i.strip() for i in match.group(1).split(",") if i.strip()]
                        )
                if "偏好景点类型:" in sys_ctx:
                    match = re.search(r"偏好景点类型:\s*(.+)", sys_ctx)
                    if match:
                        memory_attraction_types = [
                            t.strip() for t in match.group(1).split(",") if t.strip()
                        ]
                logger.debug(
                    f"记忆兴趣: {memory_interests[:5]} "
                    f"偏好类型: {memory_attraction_types}"
                )

        # ── Step 1: LLM Agent 搜索景点 ─────────────────────────────
        attractions = await self._llm_search(
            destination, num_days,
            memory_interests=memory_interests,
            memory_attraction_types=memory_attraction_types,
        )

        # ── Step 2: 并发补全坐标详情 ───────────────────────────────
        detail_tool = self.mcp_adapter.get_tool("maps_search_detail")
        if detail_tool and attractions:
            try:
                logger.debug(f"并发获取 {len(attractions)} 个 POI 详情...")
                attractions = await self._enrich_with_details(
                    detail_tool, attractions
                )
                logger.info(f"详情补全完成: {len(attractions)} 个景点")
            except Exception as e:
                logger.warning(f"详情补全失败: {e}，使用已有数据")

        # ── Step 3: 聚类分组 ────────────────────────────────────────
        logger.debug(f"按地理位置聚类，目标 {num_days} 天...")
        groups = cluster_by_location(attractions, num_days)
        logger.info(f"聚类结果: {len(groups)} 组")

        # ── Step 4: 路线优化 ────────────────────────────────────────
        daily_routes = []
        for i, group in enumerate(groups):
            optimized = optimize_route(group)
            theme = self._infer_theme(optimized)
            daily_routes.append({
                "day": i + 1,
                "attractions": optimized,
                "theme": theme,
            })
            logger.debug(f"  Day {i+1}: {theme} ({len(optimized)} 景点)")

        logger.info(f"景点规划完成: {len(daily_routes)} 天行程")

        # ── 写入记忆 ──────────────────────────────────────────────
        if self.memory_store is not None:
            sid = state.get("session_id", "default")
            uid = state.get("user_id", "default")

            # 记录到 STM
            attr_names = [a.get("name", "") for a in attractions[:10]]
            self.memory_store.record_agent_message(
                sid, "attraction",
                f"搜索完成: {len(attractions)}个景点, {len(daily_routes)}天路线",
                metadata={"attraction_names": attr_names, "count": len(attractions)},
            )

            # 写入景点特征到向量库
            if self.memory_store.long_term is not None:
                await self.memory_store.long_term.save_attraction_features(
                    attractions=attractions,
                    user_id=uid,
                    destination=destination,
                )

        return {
            "attractions": attractions,
            "daily_routes": daily_routes,
        }

    # ── LLM Agent 搜索 ──────────────────────────────────────────────
    async def _llm_search(
        self,
        destination: str,
        num_days: int,
        memory_interests: Optional[List[str]] = None,
        memory_attraction_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """让内部 LLM Agent 调用 MCP 工具搜索景点，并解析其 JSON 输出"""

        # ── 构建记忆增强的搜索提示 ────────────────────────────────
        memory_hint = ""
        if memory_interests or memory_attraction_types:
            hint_parts = ["", "## 用户历史偏好（来自记忆）", ""]
            if memory_interests:
                unique_interests = list(dict.fromkeys(memory_interests))[:8]
                hint_parts.append(f"- 历史兴趣标签: {', '.join(unique_interests)}")
            if memory_attraction_types:
                unique_types = list(dict.fromkeys(memory_attraction_types))[:5]
                hint_parts.append(f"- 偏好景点类型: {', '.join(unique_types)}")
            hint_parts.append("- 请在搜索时优先搜索与上述偏好匹配的景点类型")
            hint_parts.append("")
            memory_hint = "\n".join(hint_parts)

        task_prompt = (
            f"请帮我搜索「{destination}」的旅游景点。\n\n"
            f"行程信息：\n"
            f"- 目的地: {destination}\n"
            f"- 天数: {num_days} 天\n"
            f"- 预计需要 {num_days * 5} 个左右的候选景点\n"
            f"{memory_hint}\n"
            f"请执行以下步骤：\n"
            f"1. 使用 maps_text_search 搜索景点（可多次调用，使用不同关键词）\n"
            f"2. 对搜索结果中的每个景点，使用 maps_search_detail 获取详细信息\n"
            f"3. 根据评分和知名度筛选，保留优质景点\n"
            f"4. 按评分从高到低排序\n"
            f"5. 以 JSON 数组格式输出最终结果\n\n"
            f"请开始搜索。完成后只输出 JSON 数组，不要包含其他文字。"
        )

        try:
            logger.info(f"调用 LLM Agent 搜索景点: {destination}")
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": task_prompt}]},
                config={"configurable": {"thread_id": f"attr_search_{destination}"}},
            )

            # 从 agent 最终消息中提取 JSON
            attractions = self._extract_json_from_messages(
                result.get("messages", [])
            )

            if attractions:
                logger.info(f"LLM Agent 搜索成功: {len(attractions)} 个景点")
                return attractions
            else:
                logger.error("LLM Agent 未返回有效景点数据")
                raise RuntimeError(f"无法获取 {destination} 的景点数据，LLM Agent 未返回有效结果")

        except Exception as e:
            logger.error(f"LLM Agent 搜索失败: {e}", exc_info=True)
            raise RuntimeError(f"无法获取 {destination} 的景点数据: {e}") from e

    # ── JSON 解析 ───────────────────────────────────────────────────
    def _extract_json_from_messages(
        self, messages: List
    ) -> Optional[List[Dict[str, Any]]]:
        """从 Agent 消息列表中提取景点 JSON 数组"""

        # 倒序遍历消息，找最后一条 AI 消息
        for msg in reversed(messages):
            content = ""
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", "")

            if not content or not isinstance(content, str):
                continue

            # 尝试提取 JSON 数组
            attractions = self._parse_attraction_json(content)
            if attractions:
                return attractions

        logger.debug("未能从消息中提取到景点 JSON")
        return None

    def _parse_attraction_json(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """从文本中解析景点 JSON 数组"""

        # 策略 1: 提取 ```json ... ``` 代码块
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                if isinstance(data, list):
                    return self._normalize_attractions(data)
            except json.JSONDecodeError:
                pass

        # 策略 2: 直接找最外层 JSON 数组
        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
                if isinstance(data, list):
                    return self._normalize_attractions(data)
            except json.JSONDecodeError:
                pass

        # 策略 3: 尝试解析整个文本
        try:
            data = json.loads(text.strip())
            if isinstance(data, list):
                return self._normalize_attractions(data)
        except json.JSONDecodeError:
            pass

        return None

    def _normalize_attractions(self, raw: List[Dict]) -> List[Dict[str, Any]]:
        """标准化景点数据，补全缺失字段"""
        normalized = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            if not name:
                continue

            normalized.append({
                "id": str(item.get("id", "")),
                "name": name,
                "address": str(item.get("address", "")),
                "lat": float(item.get("lat", 0.0)),
                "lng": float(item.get("lng", 0.0)),
                "rating": float(item.get("rating", 0.0)),
                "reason": str(item.get("reason", "")),
            })
        return normalized

    # ── 并发详情补全 ────────────────────────────────────────────────
    async def _enrich_with_details(
        self, detail_tool, basic_pois: List[Dict]
    ) -> List[Dict]:
        """并发调用 maps_search_detail 获取每个 POI 的坐标和评分"""
        sem = asyncio.Semaphore(5)

        async def fetch_detail(poi: Dict) -> Dict:
            poi_id = poi.get("id", "")
            # 如果已有坐标数据且 id 为空，跳过（来自 LLM 直接输出）
            if not poi_id:
                return poi

            async with sem:
                try:
                    result = await with_retry(
                        detail_tool.ainvoke,
                        {"id": poi_id},
                        max_retries=2,
                        base_delay=0.5,
                        label=f"detail_{poi.get('name', poi_id)}",
                    )
                    # 解析 MCP 响应
                    if (
                        isinstance(result, list)
                        and len(result) > 0
                        and isinstance(result[0], dict)
                        and "text" in result[0]
                        and result[0]["text"]
                    ):
                        detail = json.loads(result[0]["text"])
                        loc = detail.get("location", "")
                        if loc:
                            parts = loc.split(",")
                            poi["lng"] = float(parts[0]) if len(parts) > 0 else poi.get("lng", 0.0)
                            poi["lat"] = float(parts[1]) if len(parts) > 1 else poi.get("lat", 0.0)
                        rating = detail.get("rating", "")
                        poi["rating"] = float(rating) if rating and rating != "" else poi.get("rating", 0.0)
                        poi["address"] = detail.get("address", "") or poi.get("address", "")
                    else:
                        logger.debug(f"POI {poi.get('name')} 详情响应为空或格式异常")
                except Exception as e:
                    logger.warning(f"获取 POI {poi.get('name', '?')} 详情失败: {e}")
                return poi

        enriched = await asyncio.gather(*[fetch_detail(p) for p in basic_pois])
        return list(enriched)

    # ── 主题推断 ────────────────────────────────────────────────────
    def _infer_theme(self, attractions: List[Dict]) -> str:
        """根据景点名称推断当日行程主题"""
        themes = {
            "历史文化": ["博物馆", "古城", "寺庙", "遗址", "故居", "陵墓", "宫殿"],
            "自然风光": ["公园", "湖", "山", "瀑布", "峡谷", "森林", "湿地"],
            "都市休闲": ["广场", "商业街", "步行街", "商圈", "夜市"],
            "亲子乐园": ["乐园", "动物园", "海洋馆", "植物园", "科技馆"],
        }
        names = " ".join([a.get("name", "") for a in attractions])
        for theme, keywords in themes.items():
            for kw in keywords:
                if kw in names:
                    return theme
        return "综合游览"

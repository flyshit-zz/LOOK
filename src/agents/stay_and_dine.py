# src/agents/stay_and_dine.py
import json
import logging
import re
from typing import Dict, Any, List, Optional

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ── 住宿餐饮推荐系统提示词 ─────────────────────────────────────────────
STAY_AND_DINE_SYSTEM_PROMPT = """你是一个专业的酒店与餐饮推荐助手。你可以调用高德地图 MCP 工具来搜索和获取酒店和餐厅信息。

## 可用工具说明
你拥有完整的高德地图 MCP 工具集，主要包括：
- **maps_text_search**: 关键字搜索 POI，支持按城市过滤。参数: keywords(关键词), city(城市), citylimit(是否限制同城)
- **maps_search_detail**: 根据 POI ID 获取详细信息，包含坐标、评分、地址等。参数: id(POI ID)
- **maps_geo**: 地理编码，将地址转换为坐标
- **maps_regeocode**: 逆地理编码，将坐标转换为地址
- **maps_direction_walking**: 步行路线规划，可估算距离和时间
- **maps_direction_driving**: 驾车路线规划
- **maps_direction_transit**: 公交路线规划

## 工作流程

### 第一步：酒店搜索
1. 分析景点分布的集中区域，确定住宿的最佳位置
2. 使用 maps_text_search 搜索酒店，关键词策略：
   - 以城市+住宿类型搜索：如 "{destination}酒店"、"{destination}{accommodation_type}"
   - 以景点集中区域搜索：如 "景点名称附近酒店"
   - 以商圈搜索：如 "{destination}市中心酒店"
3. 对搜索结果使用 maps_search_detail 获取评分、地址、电话等详情
4. 综合评价后推荐 1-2 个最优酒店

### 第二步：餐厅搜索
按早餐、午餐、晚餐三个时段分别搜索：

**早餐（breakfast）**：
- 搜索范围：酒店附近 或 出发区域
- 关键词建议： "{destination}早餐"、"早点"、"早茶"、"早餐店"
- 优先选择：快捷方便、当地特色早点、评分 ≥ 3.5

**午餐（lunch）**：
- 搜索范围：景点游玩区域附近
- 关键词建议： "{destination}特色餐厅"、"{cuisine_preference}"、"景点附近餐厅"、"午餐"
- 优先选择：有当地特色、性价比高、出餐快

**晚餐（dinner）**：
- 搜索范围：酒店附近 或 市中心美食街/商圈
- 关键词建议： "{destination}美食"、"{cuisine_preference}"、"热门餐厅"、"晚餐"
- 优先选择：环境好、评分高（≥ 4.0）、可深入体验当地美食文化

### 第三步：输出结果
汇总酒店和餐厅推荐，按指定 JSON 格式输出。

## 推荐优先级
1. **位置便利性**：距主要景点或酒店步行可达优先
2. **评分口碑**：高评分优先，酒店 ≥ 4.0，餐厅 ≥ 3.5
3. **用户偏好匹配**：严格遵循用户指定的住宿类型和饮食偏好
4. **预算合理**：在用户预算范围内推荐
5. **多样性**：兼顾不同菜系和风格

## 输出格式要求
在完成所有工具调用和分析后，你必须在最终回复中输出一个 JSON 对象，格式如下：
```json
{
  "hotels": [
    {
      "name": "酒店名称",
      "address": "详细地址",
      "rating": "4.5",
      "price_range": "300-500元/晚",
      "distance": "距市中心约2km，步行可达主要景点",
      "type": "商务酒店",
      "reason": "推荐理由（20-40字，说明为何适合此行）",
      "lat": 39.9042,
      "lng": 116.4074,
      "estimated_cost": 400
    }
  ],
  "restaurants": {
    "breakfast": [
      {
        "name": "餐厅名称",
        "address": "详细地址",
        "cuisine": "菜系类型（如粤式早茶、中式面点、西式简餐）",
        "meal_type": "breakfast",
        "rating": "4.3",
        "estimated_cost": 30,
        "reason": "推荐理由（20-40字）"
      }
    ],
    "lunch": [
      {
        "name": "餐厅名称",
        "address": "详细地址",
        "cuisine": "菜系类型",
        "meal_type": "lunch",
        "rating": "4.5",
        "estimated_cost": 60,
        "reason": "推荐理由（20-40字）"
      }
    ],
    "dinner": [
      {
        "name": "餐厅名称",
        "address": "详细地址",
        "cuisine": "菜系类型",
        "meal_type": "dinner",
        "rating": "4.7",
        "estimated_cost": 100,
        "reason": "推荐理由（20-40字）"
      }
    ]
  }
}
```

注意：
- rating 使用字符串类型
- estimated_cost 是数字，单位为「元/人」
- 酒店推荐 1-2 个，每餐推荐 1-2 个餐厅
- price_range 描述大致的价格区间
- distance 描述酒店与景点/市中心的位置关系
- 只输出 JSON 对象，不要包含任何其他文字
"""


class StayAndDineAgent(BaseAgent):
    """住宿与餐饮推荐 Agent —— 根据景点信息和用户偏好，通过 MCP 工具搜索推荐酒店和餐厅"""

    def __init__(self, mcp_adapter, llm=None, memory_store=None):
        super().__init__("stay_and_dine", "住宿餐饮推荐", memory_store=memory_store)
        self.mcp_adapter = mcp_adapter
        self.system_prompt = STAY_AND_DINE_SYSTEM_PROMPT

        # ── 提取所有 MCP 工具 ──────────────────────────────────────
        self.tools = mcp_adapter.get_tools()
        tool_names = [t.name for t in self.tools]
        logger.info(f"StayAndDineAgent 加载 {len(self.tools)} 个 MCP 工具: {tool_names}")

        # ── 初始化 LLM ─────────────────────────────────────────────
        logger.info("初始化 StayAndDineAgent LLM（DeepSeek）...")
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
        logger.info("StayAndDineAgent 内部 LLM Agent 初始化完成")

    # ── 主执行入口 ──────────────────────────────────────────────────
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        destination = state.get("destination", "")
        daily_routes = state.get("daily_routes", [])
        budget = state.get("budget")
        cuisine_preference = state.get("cuisine_preference", "")
        accommodation_preference = state.get("accommodation_preference", "hotel")
        num_people = state.get("num_people", 1)

        # ── 从记忆获取用户偏好（补充当前请求中未明确指定的偏好） ──
        memory_budget_hint = ""
        memory_cuisine_hint = ""
        memory_accom_hint = ""
        if self.memory_store is not None:
            memory_ctx = await self.get_memory_context(state)
            if memory_ctx:
                sys_ctx = memory_ctx.get("system_context", "")
                # 如果当前请求未指定偏好，从用户画像补充
                if not budget and "预算范围:" in sys_ctx:
                    match = re.search(r"预算范围:\s*(\S+)", sys_ctx)
                    if match:
                        memory_budget_hint = match.group(1)
                        logger.debug(f"从记忆获取预算范围: {memory_budget_hint}")
                if not cuisine_preference and "饮食偏好:" in sys_ctx:
                    match = re.search(r"饮食偏好:\s*(\S+)", sys_ctx)
                    if match:
                        memory_cuisine_hint = match.group(1)
                        logger.debug(f"从记忆获取饮食偏好: {memory_cuisine_hint}")
                if accommodation_preference == "hotel" and "旅行风格:" in sys_ctx:
                    # 旅行风格可能暗示住宿偏好
                    match = re.search(r"旅行风格:\s*(\S+)", sys_ctx)
                    if match:
                        style = match.group(1)
                        if style == "luxury":
                            memory_accom_hint = "hotel"
                        elif style == "budget":
                            memory_accom_hint = "hostel"

        logger.info(
            f"开始住宿餐饮推荐: dest={destination} "
            f"routes={len(daily_routes)}天 budget={budget} "
            f"cuisine={cuisine_preference} accom={accommodation_preference}"
        )

        # ── 构建任务提示 ─────────────────────────────────────────
        task_prompt = self._build_task_prompt(
            destination=destination,
            daily_routes=daily_routes,
            budget=budget,
            cuisine_preference=cuisine_preference or memory_cuisine_hint,
            accommodation_preference=accommodation_preference if accommodation_preference != "hotel" or not memory_accom_hint else memory_accom_hint,
            num_people=num_people,
            memory_budget_hint=memory_budget_hint,
            memory_cuisine_hint=memory_cuisine_hint,
        )

        # ── 调用 LLM Agent 搜索 ──────────────────────────────────
        try:
            logger.info(f"调用 LLM Agent 搜索酒店和餐厅: {destination}")
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": task_prompt}]},
                config={"configurable": {"thread_id": f"sad_{destination}"}},
            )

            data = self._extract_json_from_messages(result.get("messages", []))

            if data:
                hotels = data.get("hotels", [])
                restaurants = data.get("restaurants", {})
                breakfast_count = len(restaurants.get("breakfast", []))
                lunch_count = len(restaurants.get("lunch", []))
                dinner_count = len(restaurants.get("dinner", []))
                total_restaurants = breakfast_count + lunch_count + dinner_count
                logger.info(
                    f"推荐完成: {len(hotels)}个酒店, "
                    f"早餐{breakfast_count} 午餐{lunch_count} 晚餐{dinner_count}"
                )

                # ── 记录产出到 STM ──────────────────────────────
                if self.memory_store is not None:
                    sid = state.get("session_id", "default")
                    self.memory_store.record_agent_message(
                        sid, "stay_and_dine",
                        f"推荐完成: {len(hotels)}个酒店, {total_restaurants}个餐厅",
                        metadata={
                            "hotel_count": len(hotels),
                            "restaurant_count": total_restaurants,
                        },
                    )

                return {
                    "hotels": hotels,
                    "restaurants": restaurants,
                }
            else:
                logger.error("LLM Agent 未返回有效的酒店/餐厅数据")
                return {"hotels": [], "restaurants": {}}

        except Exception as e:
            logger.error(f"住宿餐饮推荐失败: {e}", exc_info=True)
            return {"hotels": [], "restaurants": {}}

    # ── 构建任务提示词 ──────────────────────────────────────────────
    def _build_task_prompt(
        self,
        destination: str,
        daily_routes: List[Dict],
        budget: Optional[float],
        cuisine_preference: str,
        accommodation_preference: str,
        num_people: int,
        memory_budget_hint: str = "",
        memory_cuisine_hint: str = "",
    ) -> str:
        """根据景点行程和用户偏好构建详细的任务提示"""

        parts = [
            f"请帮我为「{destination}」的旅行推荐酒店和餐厅。",
            "",
        ]

        # ── 预算和人数组 ──────────────────────────────────────────
        if budget:
            per_person = budget / num_people if num_people > 0 else budget
            parts.append(f"总预算: {budget}元（{num_people}人，人均约{per_person:.0f}元）")
        elif memory_budget_hint:
            parts.append(f"预算参考（来自用户历史偏好）: {memory_budget_hint}")
        else:
            parts.append(f"人数: {num_people}人（预算未限定）")

        # ── 住宿偏好 ──────────────────────────────────────────────
        accom_map = {
            "hotel": "酒店（优先中高档商务酒店或度假酒店）",
            "hostel": "青年旅舍/民宿（优先性价比高的特色民宿）",
            "apartment": "公寓/民宿（优先舒适且可短租的公寓式住宿）",
        }
        accom_text = accom_map.get(accommodation_preference, "酒店")
        parts.append(f"住宿偏好: {accom_text}")

        # ── 饮食偏好 ──────────────────────────────────────────────
        if cuisine_preference:
            parts.append(f"饮食偏好: {cuisine_preference}")
        elif memory_cuisine_hint:
            parts.append(f"饮食偏好（来自用户历史偏好）: {memory_cuisine_hint}")
        else:
            parts.append("饮食偏好: 无特殊要求，优先推荐当地特色美食和热门餐厅")

        # ── 景点行程上下文 ────────────────────────────────────────
        parts.append("")
        parts.append("── 已规划的每日景点行程 ──")
        parts.append("请基于景点分布选择最便利的酒店位置，并根据每日游玩路线推荐就近餐厅。")
        parts.append("")

        for route in daily_routes:
            day = route.get("day", "?")
            theme = route.get("theme", "综合游览")
            parts.append(f"## 第{day}天 ── {theme}")
            for attr in route.get("attractions", []):
                name = attr.get("name", "未知景点")
                address = attr.get("address", "")
                lat = attr.get("lat", 0)
                lng = attr.get("lng", 0)
                rating = attr.get("rating", 0)
                reason = attr.get("reason", "")
                coord_str = f"坐标({lat}, {lng})" if lat and lng else ""
                rating_str = f"评分{rating}" if rating else ""
                meta = " | ".join(filter(None, [address, coord_str, rating_str]))
                line = f"  - {name}"
                if meta:
                    line += f" | {meta}"
                if reason:
                    line += f" | {reason}"
                parts.append(line)

        # ── 推荐要求 ──────────────────────────────────────────────
        parts.append("")
        parts.append("── 推荐要求 ──")
        parts.append("1. **酒店**（1-2个）：优先位于主要景点集中区域或交通便利的市中心")
        parts.append("2. **早餐**（1-2个）：靠近酒店或第1天出发区域，快捷实惠，适合出发前用餐")
        parts.append("3. **午餐**（1-2个）：靠近当天上午游玩景点附近，有当地特色，性价比高")
        parts.append("4. **晚餐**（1-2个）：靠近酒店或市中心商圈，环境好评分高，适合放松享用")
        parts.append("")
        parts.append("请按以下步骤执行：")
        parts.append("1. 分析景点分布，确定酒店最佳区域 → 使用 maps_text_search 搜索酒店")
        parts.append("2. 获取酒店详情 → 使用 maps_search_detail")
        parts.append("3. 搜索早餐/午餐/晚餐餐厅 → 使用 maps_text_search")
        parts.append("4. 获取餐厅详情 → 使用 maps_search_detail")
        parts.append("5. 筛选最优选项，按 JSON 格式输出")
        parts.append("")
        parts.append("完成后只输出 JSON 对象，不要包含其他文字。")

        return "\n".join(parts)

    # ── JSON 解析 ───────────────────────────────────────────────────
    def _extract_json_from_messages(
        self, messages: List
    ) -> Optional[Dict[str, Any]]:
        """从 Agent 消息列表中提取酒店/餐厅 JSON 对象"""

        # 倒序遍历消息，找最后一条 AI 消息
        for msg in reversed(messages):
            content = ""
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", "")

            if not content or not isinstance(content, str):
                continue

            data = self._parse_result_json(content)
            if data:
                return data

        logger.debug("未能从消息中提取到酒店/餐厅 JSON")
        return None

    def _parse_result_json(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本中解析包含 hotels 和 restaurants 的 JSON 对象"""

        # 策略 1: 提取 ```json ... ``` 代码块
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                if isinstance(data, dict) and ("hotels" in data or "restaurants" in data):
                    return self._normalize_data(data)
            except json.JSONDecodeError:
                pass

        # 策略 2: 正则匹配最外层 JSON 对象
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                if isinstance(data, dict) and ("hotels" in data or "restaurants" in data):
                    return self._normalize_data(data)
            except json.JSONDecodeError:
                pass

        # 策略 3: 尝试解析整个文本
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and ("hotels" in data or "restaurants" in data):
                return self._normalize_data(data)
        except json.JSONDecodeError:
            pass

        return None

    def _normalize_data(self, raw: Dict) -> Dict[str, Any]:
        """标准化酒店和餐厅数据，补全缺失字段"""

        # ── 标准化酒店 ──────────────────────────────────────────────
        hotels = raw.get("hotels", [])
        normalized_hotels = []
        for h in hotels:
            if not isinstance(h, dict):
                continue
            name = h.get("name", "")
            if not name:
                continue

            normalized_hotels.append({
                "name": str(name),
                "address": str(h.get("address", "")),
                "rating": str(h.get("rating", "")),
                "price_range": str(h.get("price_range", "")),
                "distance": str(h.get("distance", "")),
                "type": str(h.get("type", "")),
                "reason": str(h.get("reason", "")),
                "lat": float(h.get("lat", 0.0)),
                "lng": float(h.get("lng", 0.0)),
                "estimated_cost": int(h.get("estimated_cost", 0)),
            })

        # ── 标准化餐厅 ──────────────────────────────────────────────
        restaurants = raw.get("restaurants", {})
        normalized_restaurants = {}

        for meal_type in ("breakfast", "lunch", "dinner"):
            meal_list = restaurants.get(meal_type, [])
            if not isinstance(meal_list, list):
                meal_list = []
            normalized_meal = []
            for r in meal_list:
                if not isinstance(r, dict):
                    continue
                r_name = r.get("name", "")
                if not r_name:
                    continue

                normalized_meal.append({
                    "name": str(r_name),
                    "address": str(r.get("address", "")),
                    "cuisine": str(r.get("cuisine", "")),
                    "meal_type": meal_type,
                    "rating": str(r.get("rating", "")),
                    "estimated_cost": int(r.get("estimated_cost", 0)),
                    "reason": str(r.get("reason", "")),
                })
            normalized_restaurants[meal_type] = normalized_meal

        return {
            "hotels": normalized_hotels,
            "restaurants": normalized_restaurants,
        }

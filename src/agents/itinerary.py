# src/agents/itinerary.py
import logging
from typing import Dict, Any, List

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

# ── LLM 结构化输出 ──────────────────────────────────────────────────────
class ItineraryOutput(BaseModel):
    itinerary: str = Field(description="完整的 Markdown 格式旅行行程")


# ── 系统提示词 ──────────────────────────────────────────────────────────
ITINERARY_SYSTEM_PROMPT = """你是一个专业的旅行行程规划师。你的任务是根据提供的景点、酒店、餐厅数据，生成一份合理、实用的 Markdown 格式旅行行程。

## 核心原则

1. **只使用提供的数据**：景点、酒店、餐厅只能从输入数据中选取，不得编造任何不存在的景点、酒店或餐厅。
2. **不编造价格/费用**：不要估算任何门票价格、酒店价格、餐饮费用。如果数据中有价格字段可以使用，没有则不要编造。
3. **不提及天气**：不要讨论天气、温度、穿衣建议等内容。
4. **合理安排时间**：根据旅行风格合理分配每天的上下午景点，考虑景点间的距离和游览时长。
5. **匹配餐厅到天**：将早餐/午餐/晚餐餐厅合理分配到具体某一天，优先考虑位置就近。

## 行程结构

输出 Markdown 格式，包含以下部分：

### 1. 标题与概览
```
# 🗺️ {目的地} 旅行计划
简要介绍行程概况（天数、风格、亮点）
```

### 2. 住宿推荐（如有酒店数据）
列出推荐的酒店，包含名称、地址、评分、推荐理由（从数据中提取，不编造）。

### 3. 每日行程（重点）
每天按以下结构组织：

```
## 📅 第 N 天：{主题}

### 🌅 上午（08:00 — 12:00）
- 🍳 早餐：{餐厅名称}（从早餐数据中选）
- 🎯 景点1：{名称} — {地址}，{推荐理由}
- 🎯 景点2：...

### 🍜 午餐（12:00 — 13:30）
- {餐厅名称}（从午餐数据中选）

### 🏛️ 下午（13:30 — 17:30）
- 🎯 景点3：...
- 🎯 景点4：...

### 🌆 晚餐（18:00 — 20:00）
- {餐厅名称}（从晚餐数据中选）

### 🌙 晚间建议
根据当天主题给 1 句晚间活动建议。
```

### 4. 旅行贴士
给出 3-5 条通用旅行建议（如提前预约、交通出行、常备药品等），不要涉及天气和预算。

## 时间分配策略

根据旅行风格分配景点：
- **relaxed（轻松）**：每天上午 1-2 个景点，下午 1 个景点，留充足自由时间
- **balanced（均衡）**：上下午均匀分配
- **intensive（紧凑）**：尽可能多安排，但要合理（考虑景点间交通）

## 餐厅分配策略

- 如果某餐别有 N 个餐厅，天数为 D：
  - N >= D：每天分配不同的餐厅
  - N < D：循环分配，同一天同一餐只推荐 1 个
- 优先根据餐厅地址与当日景点区域的近远来匹配

## 输出要求

- 只输出 Markdown 文本，不要额外解释
- 使用丰富的 emoji 增强可读性
- 每天之间用 `---` 分隔
- 不要留空占位符——如果某餐没有数据，写"就近自由选择"即可
- 行程末尾加一句祝福语"""


class ItineraryAgent(BaseAgent):
    """行程生成 Agent —— 使用 LLM 智能整合多源数据生成合理行程"""

    def __init__(self, llm=None, memory_store=None):
        super().__init__("itinerary", "行程生成", memory_store=memory_store)
        logger.info("初始化 ItineraryAgent，加载 DeepSeek 模型...")
        self.llm = llm or init_chat_model(
            model="deepseek-chat",
            temperature=0.3,
        )
        self.structured_llm = self.llm.with_structured_output(ItineraryOutput)
        logger.info("ItineraryAgent 初始化完成")

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        destination = state.get("destination", "")
        daily_routes = state.get("daily_routes", [])
        hotels = state.get("hotels", [])
        restaurants = state.get("restaurants", {})
        num_days = state.get("num_days", len(daily_routes))
        travel_style = state.get("travel_style", "balanced")
        interests = state.get("interests", [])

        logger.info(
            f"LLM 生成行程: dest={destination} days={len(daily_routes)} "
            f"hotels={len(hotels)} restaurants={len(restaurants)} style={travel_style}"
        )

        # ── 构建结构化数据摘要给 LLM ──────────────────────────────────
        prompt = self._build_prompt(
            destination=destination,
            daily_routes=daily_routes,
            hotels=hotels,
            restaurants=restaurants,
            num_days=num_days,
            travel_style=travel_style,
            interests=interests,
        )

        # ── 调用 LLM 生成行程 ────────────────────────────────────────
        try:
            result = await with_retry(
                self.structured_llm.ainvoke,
                prompt,
                max_retries=2,
                base_delay=1.0,
                label="itinerary_llm",
            )
            itinerary = result.itinerary
            logger.info(f"LLM 行程生成完成: {len(itinerary)} 字符")
        except Exception as e:
            logger.warning(f"LLM 行程生成失败: {e}，启用模板兜底")
            itinerary = self._fallback_itinerary(
                destination, daily_routes, hotels, restaurants
            )

        # ── 记录行程产出到 STM ────────────────────────────────────
        if self.memory_store is not None:
            sid = state.get("session_id", "default")
            self.memory_store.record_agent_message(
                sid, "itinerary",
                f"行程已生成: {destination}{num_days}日游 ({len(itinerary)}字符)",
                metadata={
                    "destination": destination,
                    "num_days": num_days,
                    "itinerary_length": len(itinerary),
                },
            )

        return {
            "itinerary": itinerary,
            "messages": [{"role": "assistant", "content": itinerary}],
        }

    # ── Prompt 构建 ────────────────────────────────────────────────────
    def _build_prompt(
        self,
        destination: str,
        daily_routes: List[Dict],
        hotels: List[Dict],
        restaurants: Dict[str, List],
        num_days: int,
        travel_style: str,
        interests: List[str],
    ) -> str:
        """构建 LLM 输入：将结构化数据序列化为易读文本"""
        parts = [ITINERARY_SYSTEM_PROMPT, "", "── 输入数据 ──", ""]

        # 基本信息
        parts.append(f"目的地: {destination}")
        parts.append(f"天数: {num_days}")
        parts.append(f"旅行风格: {travel_style}")
        if interests:
            parts.append(f"兴趣偏好: {', '.join(interests)}")
        parts.append("")

        # ── 酒店数据 ──────────────────────────────────────────────────
        if hotels:
            parts.append("## 可选酒店")
            parts.append("")
            for i, h in enumerate(hotels, 1):
                parts.append(f"{i}. {h.get('name', '?')}")
                if h.get("address"):
                    parts.append(f"   地址: {h['address']}")
                if h.get("rating"):
                    parts.append(f"   评分: {h['rating']}")
                if h.get("price_range"):
                    parts.append(f"   价格区间: {h['price_range']}")
                if h.get("type"):
                    parts.append(f"   类型: {h['type']}")
                if h.get("distance"):
                    parts.append(f"   位置: {h['distance']}")
                if h.get("reason"):
                    parts.append(f"   推荐理由: {h['reason']}")
            parts.append("")

        # ── 每日景点 ──────────────────────────────────────────────────
        parts.append("## 每日景点路线")
        parts.append("")
        for route in daily_routes:
            day = route.get("day", "?")
            theme = route.get("theme", "综合游览")
            parts.append(f"### 第{day}天: {theme}")
            for attr in route.get("attractions", []):
                parts.append(f"- {attr.get('name', '?')}")
                if attr.get("address"):
                    parts.append(f"  地址: {attr['address']}")
                if attr.get("rating"):
                    parts.append(f"  评分: {attr['rating']}")
                if attr.get("reason"):
                    parts.append(f"  推荐理由: {attr['reason']}")
            parts.append("")

        # ── 餐厅数据 ──────────────────────────────────────────────────
        if restaurants:
            parts.append("## 可选餐厅")
            parts.append("")
            meal_labels = {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐"}
            for meal_type in ("breakfast", "lunch", "dinner"):
                meal_list = restaurants.get(meal_type, [])
                if meal_list:
                    parts.append(f"### {meal_labels[meal_type]}（共 {len(meal_list)} 个）")
                    for r in meal_list:
                        parts.append(f"- {r.get('name', '?')}")
                        if r.get("cuisine"):
                            parts.append(f"  菜系: {r['cuisine']}")
                        if r.get("address"):
                            parts.append(f"  地址: {r['address']}")
                        if r.get("rating"):
                            parts.append(f"  评分: {r['rating']}")
                        if r.get("reason"):
                            parts.append(f"  推荐理由: {r['reason']}")
                    parts.append("")

        parts.append("── 请根据以上数据生成完整行程 Markdown ──")

        return "\n".join(parts)

    # ── 兜底模板 ──────────────────────────────────────────────────────
    def _fallback_itinerary(
        self,
        destination: str,
        daily_routes: List[Dict],
        hotels: List[Dict],
        restaurants: Dict[str, List],
    ) -> str:
        """LLM 失败时的纯模板兜底，保证不报错"""
        lines = [
            f"# 🗺️ {destination} 旅行计划",
            "",
            f"**{len(daily_routes)}** 天行程，以下为每日安排：",
            "",
        ]

        if hotels:
            lines.append("## 🏨 住宿推荐")
            lines.append("")
            for h in hotels:
                name = h.get("name", "")
                addr = h.get("address", "")
                rating = h.get("rating", "")
                reason = h.get("reason", "")
                if name:
                    lines.append(f"- **{name}**")
                    if addr:
                        lines.append(f"  📍 {addr}")
                    if rating:
                        lines.append(f"  ⭐ {rating}")
                    if reason:
                        lines.append(f"  > {reason}")
            lines.append("")
            lines.append("---")
            lines.append("")

        for day in daily_routes:
            day_num = day.get("day", 0)
            theme = day.get("theme", "综合游览")
            attractions = day.get("attractions", [])

            lines.append(f"## 📅 第 {day_num} 天：{theme}")
            lines.append("")

            mid = len(attractions) // 2 or 1
            morning = attractions[:mid]
            afternoon = attractions[mid:]

            lines.append("### 🌅 上午")
            for attr in morning:
                name = attr.get("name", "未命名")
                addr = attr.get("address", "")
                rating = attr.get("rating", 0)
                lines.append(f"- 🎯 **{name}**")
                if addr:
                    lines.append(f"  📍 {addr}")
                if rating:
                    lines.append(f"  ⭐ {rating}")
            lines.append("")

            lines.append("### 🍜 午餐")
            lines.append("就近自由选择")
            lines.append("")

            lines.append("### 🏛️ 下午")
            for attr in afternoon:
                name = attr.get("name", "未命名")
                addr = attr.get("address", "")
                rating = attr.get("rating", 0)
                lines.append(f"- 🎯 **{name}**")
                if addr:
                    lines.append(f"  📍 {addr}")
                if rating:
                    lines.append(f"  ⭐ {rating}")
            lines.append("")

            lines.append("### 🌆 晚餐")
            lines.append("就近自由选择")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

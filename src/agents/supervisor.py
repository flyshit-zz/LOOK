# src/agents/supervisor.py
import logging
from typing import Dict, Any, Optional
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from src.agents.base import BaseAgent
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="决策推理过程，说明为什么选择下一步")
    next_agent: str = Field(description="下一步Agent: attraction, stay_and_dine, itinerary, end")
    response_to_user: str = Field(default="", description="显示给用户的进度提示")
    destination: str = Field(default="", description="提取或确认的目的地城市")
    num_days: int = Field(default=3, description="行程天数")


# ── Supervisor 系统提示词 ─────────────────────────────────────────────
SUPERVISOR_SYSTEM_PROMPT = """你是一个旅行规划调度助手。根据当前对话状态，决定下一步操作。

路由规则：
- 用户输入中没有明确目的地 → next_agent="attraction"，尝试提取目的地
- 已确认目的地但还没搜景点 → next_agent="attraction"
- 已有景点列表但还没推荐酒店和餐厅 → next_agent="stay_and_dine"
- 已有酒店和餐厅但还没生成最终行程 → next_agent="itinerary"
- 行程已生成完整 → next_agent="end"

要求：
1. reasoning 字段：用中文简述你的判断依据
2. response_to_user 字段：给用户的中文进度提示（如"正在为您搜索北京的景点…"）
3. 如果当前是 end，response_to_user 应总结已完成的工作
4. 若尚未提取目的地，填写 destination 和 num_days；否则这两项可留空"""


class SupervisorAgent(BaseAgent):
    def __init__(self, llm=None, memory_store=None):
        super().__init__("supervisor", "任务调度", memory_store=memory_store)
        logger.info("初始化 SupervisorAgent，加载 DeepSeek 模型...")
        self.llm = llm or init_chat_model(
            model="deepseek-chat",
            temperature=0.2,
        )
        self.structured_llm = self.llm.with_structured_output(SupervisorDecision)
        logger.info("SupervisorAgent 初始化完成")

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        has_attractions = bool(state.get("attractions"))
        has_hotels = "hotels" in state
        has_itinerary = bool(state.get("itinerary"))
        destination = state.get("destination", "")

        logger.debug(
            f"Supervisor 执行: input={user_input[:40]}... "
            f"dest={destination} has_attr={has_attractions} has_hotels={has_hotels} has_it={has_itinerary}"
        )

        # ── 获取记忆上下文 ───────────────────────────────────────────
        memory_context = state.get("memory_context")
        if memory_context is None and self.memory_store is not None:
            memory_context = await self.get_memory_context(state)

        # ── 构建带上下文的提示词 ─────────────────────────────────────
        prompt = self._build_prompt(
            user_input=user_input,
            destination=destination,
            has_attractions=has_attractions,
            has_hotels=has_hotels,
            has_itinerary=has_itinerary,
            state=state,
            memory_context=memory_context,
        )

        # ── 调用 LLM（带重试） ────────────────────────────────────────
        try:
            decision = await with_retry(
                self.structured_llm.ainvoke,
                prompt,
                max_retries=3,
                base_delay=1.0,
                label="supervisor_llm",
            )
            logger.info(
                f"LLM 决策: next={decision.next_agent} "
                f"reason={decision.reasoning[:60]}..."
            )
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}，启用兜底路由")
            decision = self._fallback_routing(
                destination, has_attractions, has_hotels, has_itinerary
            )

        # ── 回写记忆（STM） ───────────────────────────────────────────
        if self.memory_store is not None:
            sid = state.get("session_id", "default")
            # 记录 Agent 产出消息
            if decision.response_to_user:
                self.memory_store.record_agent_message(
                    sid, "supervisor", decision.response_to_user,
                    metadata={"next_agent": decision.next_agent},
                )
            # 记录关键决策
            if decision.destination:
                self.memory_store.record_decision(
                    sid, "destination", decision.destination,
                    note=f"提取目的地: {decision.destination}",
                )
            self.memory_store.record_decision(
                sid, "routing", decision.next_agent,
                note=decision.reasoning[:200],
            )

        # ── 构建返回的状态更新 ───────────────────────────────────────
        result: Dict[str, Any] = {"next_agent": decision.next_agent}

        # 提取/更新目的地
        if decision.destination and not destination:
            result["destination"] = decision.destination
            result["num_days"] = decision.num_days
            logger.info(f"目的地提取: {decision.destination} ({decision.num_days}天)")

        # reasoning 记录到日志（已在上面记录，这里只做结构化留存）
        if decision.reasoning:
            result["supervisor_reasoning"] = decision.reasoning

        # response_to_user 推入消息列表
        if decision.response_to_user:
            result["messages"] = [
                {"role": "assistant", "content": decision.response_to_user}
            ]

        # 行程完成时的收尾
        if decision.next_agent == "end":
            result["is_complete"] = True
            itinerary = state.get("itinerary", "")
            if itinerary:
                result["messages"] = [
                    {"role": "assistant", "content": itinerary}
                ]

        return result

    # ── 私有方法 ──────────────────────────────────────────────────────

    def _build_prompt(
        self,
        user_input: str,
        destination: str,
        has_attractions: bool,
        has_hotels: bool,
        has_itinerary: bool,
        state: Dict[str, Any],
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建包含上下文的 prompt，让 LLM 做出更明智的决策"""
        parts = [SUPERVISOR_SYSTEM_PROMPT, "", "── 当前状态 ──"]

        parts.append(f"用户输入: {user_input}")

        if destination:
            parts.append(f"目的地: {destination}")
        else:
            parts.append("目的地: 未提取")

        if has_attractions:
            attr_count = len(state.get("attractions", []))
            routes_count = len(state.get("daily_routes", []))
            parts.append(f"景点: 已搜索 ({attr_count} 个景点, {routes_count} 天路线)")
        else:
            parts.append("景点: 未搜索")

        if has_hotels:
            hotel_count = len(state.get("hotels", []))
            restaurants = state.get("restaurants", {})
            breakfast_count = len(restaurants.get("breakfast", []))
            lunch_count = len(restaurants.get("lunch", []))
            dinner_count = len(restaurants.get("dinner", []))
            parts.append(
                f"酒店和餐厅: 已推荐 ({hotel_count}个酒店, "
                f"早餐{breakfast_count} 午餐{lunch_count} 晚餐{dinner_count})"
            )
        else:
            parts.append("酒店和餐厅: 未推荐")

        if has_itinerary:
            it = state.get("itinerary", "")
            parts.append(f"行程: 已生成 ({len(it)} 字符)")
        else:
            parts.append("行程: 未生成")

        # ── 注入记忆上下文 ─────────────────────────────────────────
        if memory_context:
            sys_ctx = memory_context.get("system_context", "")
            if sys_ctx:
                parts.append("")
                parts.append("── 用户记忆 ──")
                parts.append(sys_ctx)

            relevant = memory_context.get("relevant_memories", [])
            if relevant:
                parts.append("")
                parts.append("── 相关历史行程 ──")
                for mem in relevant[:3]:
                    parts.append(f"- {mem.get('content', '')[:200]}")

        parts.append("")
        parts.append("请决定下一步操作（返回 JSON）。")

        return "\n".join(parts)

    def _fallback_routing(
        self,
        destination: str,
        has_attractions: bool,
        has_hotels: bool,
        has_itinerary: bool,
    ) -> SupervisorDecision:
        """纯规则兜底路由——当 LLM 调用失败时使用"""
        if not destination:
            return SupervisorDecision(
                reasoning="LLM 不可用，规则判断：缺少目的地",
                next_agent="attraction",
                response_to_user="正在分析您的需求...",
                destination="",
                num_days=3,
            )
        if not has_attractions:
            return SupervisorDecision(
                reasoning="LLM 不可用，规则判断：需要搜索景点",
                next_agent="attraction",
                response_to_user=f"正在为您搜索{destination}的景点...",
            )
        if not has_hotels:
            return SupervisorDecision(
                reasoning="LLM 不可用，规则判断：需要推荐酒店和餐厅",
                next_agent="stay_and_dine",
                response_to_user=f"景点已就绪，正在为您推荐{destination}的酒店和餐厅...",
            )
        if not has_itinerary:
            return SupervisorDecision(
                reasoning="LLM 不可用，规则判断：需要生成行程",
                next_agent="itinerary",
                response_to_user="酒店和餐厅已推荐，正在生成最终行程方案...",
            )
        return SupervisorDecision(
            reasoning="LLM 不可用，规则判断：任务完成",
            next_agent="end",
            response_to_user="行程已生成完毕！",
        )

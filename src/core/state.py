from typing import TypedDict, Annotated,List,Optional,Any,Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from datetime import datetime
from typing import Dict

class TravelState(TypedDict, total=False):
    """
     旅行助手的全局状态
    """
#-------------对话相关
    messages:Annotated[List[BaseMessage],add_messages]
    user_input:str
    user_id: Optional[str]
    session_id:Optional[str]
    memory_context: Optional[Dict[str, Any]]
    """记忆模块组装的上下文（由 MemoryStore.get_context 填充）"""
#----------- 用户偏好
    destination: Optional[str]
    """目的地城市名称"""
    start_date: Optional[datetime]
    """行程开始日期"""
    end_date: Optional[datetime]
    """行程结束日期"""
    num_days: Optional[int]
    """行程天数（可由日期计算得出）"""
    budget: Optional[float]
    """总预算（单位：元）"""
    num_people: Optional[int]
    """同行人数"""
    interests: List[str]
    """用户兴趣标签列表（如 ['历史文化', '美食']）"""
    cuisine_preference: Optional[str]
    """饮食偏好（如 '川菜', '素食'）"""
    accommodation_preference: Optional[Literal['hotel', 'hostel', 'apartment']]
    """住宿偏好类型"""
    travel_style: Optional[Literal['relaxed', 'balanced', 'intensive']]
    """旅行风格：轻松/均衡/紧凑"""
    # ==================== 各Agent产出数据 ====================
    # recommended_destinations: List[Dict[str, Any]]
    # """目的地推荐Agent产出：候选目的地列表"""
    
    attractions: List[Dict[str, Any]]
    """景点规划Agent产出：所有景点详情列表"""
    
    daily_routes: List[Dict[str, Any]]
    """景点规划Agent产出：每日路线（已聚类和排序）"""
    
    hotels: List[Dict[str, Any]]
    """酒店推荐Agent产出：酒店列表"""
    
    restaurants: List[Dict[str, Any]]
    """餐厅推荐Agent产出：餐厅列表"""
    
    # weather_info: Dict[str, Any]
    # """天气Agent产出：天气信息（当前+预报）"""
    
    # budget_analysis: Dict[str, Any]
    # """预算分析Agent产出：费用估算明细"""
    
    itinerary: str
    """行程生成Agent产出：最终完整行程（Markdown 文本）"""
    
    # ==================== 流程控制 ====================
    current_agent: str
    """当前正在执行的Agent名称"""
    
    next_agent: Optional[str]
    """由Supervisor决策的下一个Agent名称，优先于路由逻辑"""
    
    is_complete: bool
    """是否已完成所有任务"""
    
    error: Optional[str]
    """错误信息（如果有）"""
    
    retry_count: int
    """当前重试次数（用于错误恢复）"""
    
    # ==================== 元数据 ====================
    execution_id: Optional[str]
    """本次执行的唯一ID"""
    
    started_at: Optional[datetime]
    """开始时间"""
    
    completed_at: Optional[datetime]
    """完成时间"""
    @classmethod
    def create_empty(cls, user_input: str = "", user_id: str = "", session_id: str = "") -> 'TravelState':
        """创建一个空的状态实例（用于初始化）"""
        return {
            "messages": [],
            "user_input": user_input,
            "user_id": user_id,
            "session_id": session_id,
            "interests": [],
            "is_complete": False,
            "retry_count": 0,
        }


# def ensure_state_valid(state: TravelState) -> None:
#     """
#     检查状态是否包含必要字段（在关键节点前调用）
#     若缺少必要字段，抛出 StateError
#     """
#     # 至少需要用户输入或目的地
#     if not state.get("user_input") and not state.get("destination"):
#         raise StateError("状态中缺少 user_input 或 destination")
    
#     # 如果已有目的地，检查日期是否完整
#     if state.get("destination") and (not state.get("start_date") or not state.get("end_date")):
#         # 允许未提供日期，但会使用默认值
#         pass
    
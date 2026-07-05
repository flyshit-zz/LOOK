"""自定义异常类 - 用于整个系统的错误处理"""


class TravelAssistantError(Exception):
    """所有自定义异常的基类"""
    pass


class StateError(TravelAssistantError):
    """状态管理相关错误（如缺少必要字段）"""
    pass


class AgentError(TravelAssistantError):
    """Agent执行过程中的错误"""
    pass


class AgentNotFoundError(AgentError):
    """找不到指定的Agent"""
    pass


class AgentExecutionError(AgentError):
    """Agent执行时发生异常"""
    def __init__(self, agent_name: str, message: str, original_error: Exception = None):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}' 执行失败: {message}")


class MCPError(TravelAssistantError):
    """MCP工具调用相关错误"""
    pass


class MCPToolNotFoundError(MCPError):
    """找不到指定的MCP工具"""
    pass


class MCPConnectionError(MCPError):
    """MCP Server连接失败"""
    pass


class RoutingError(TravelAssistantError):
    """路由逻辑错误（如无法决定下一步）"""
    pass


class CheckpointError(TravelAssistantError):
    """检查点保存/恢复失败"""
    pass


class ValidationError(TravelAssistantError):
    """数据验证失败（如输入参数格式错误）"""
    pass
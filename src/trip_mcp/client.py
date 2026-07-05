"""MCP客户端管理器"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Dict, Any, List, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP客户端管理器"""
    
    def __init__(self, server_configs: Dict[str, Dict[str, Any]]):
        self.server_configs = server_configs
        self.client: Optional[MultiServerMCPClient] = None
        self._tools: Optional[List[BaseTool]] = None
        self._initialized = False
    
    async def initialize(self) -> "MCPClient":
        """初始化客户端"""
        if self._initialized:
            return self
        
        logger.info("初始化MCP客户端，连接服务器: %s", list(self.server_configs.keys()))
        self.client = MultiServerMCPClient(self.server_configs)
        self._tools = await self.client.get_tools()
        self._initialized = True
        logger.info("MCP客户端初始化成功，加载 %d 个工具", len(self._tools))
        return self
    
    async def get_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        if not self._initialized:
            await self.initialize()
        return self._tools or []
    
    async def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        tools = await self.get_tools()
        for tool in tools:
            if tool.name == name:
                return tool
        return None
    
    async def close(self) -> None:
        """关闭连接"""
        if self.client:
            logger.info("关闭MCP客户端")
            self._initialized = False
            self._tools = None


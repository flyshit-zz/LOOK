"""MCP工具适配器"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Dict
from langchain_core.tools import BaseTool
from src.trip_mcp.client import MCPClient
import logging

logger = logging.getLogger(__name__)


class MCPAdapter:
    """MCP工具适配器"""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self._tools: Optional[List[BaseTool]] = None
    
    async def initialize(self) -> "MCPAdapter":
        """初始化适配器"""
        if self._tools is None:
            await self.client.initialize()
            self._tools = await self.client.get_tools()
            logger.info("MCP适配器初始化完成，工具数量: %d", len(self._tools))
        return self
    
    def get_tools(self) -> List[BaseTool]:
        """获取所有工具（同步）"""
        if self._tools is None:
            raise RuntimeError("适配器未初始化，请先调用 initialize()")
        return self._tools
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取单个工具"""
        for tool in self.get_tools():
            if tool.name == name:
                return tool
        return None
    
    def to_tool_dict(self) -> Dict[str, BaseTool]:
        """转为字典"""
        return {tool.name: tool for tool in self.get_tools()}


if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=" * 50)
        print("MCP适配器测试")
        print("=" * 50)
        
        from src.trip_mcp.registry import MCPServerRegistry
        
        # 使用真实配置
        registry = MCPServerRegistry()
        config = registry.load_config()
        
        client = MCPClient(config)
        adapter = MCPAdapter(client)
        await adapter.initialize()
        
        tools = adapter.get_tools()
        print(f"\n加载工具数量: {len(tools)}")
        
        # 显示工具名称
        for i, tool in enumerate(tools, 1):
            print(f"  {i}. {tool.name}")
        
        print("\n[OK] 适配器初始化成功!")
        print("=" * 50)
    
    asyncio.run(test())
"""MCP服务注册与发现"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class MCPServerRegistry:
    """MCP服务注册中心"""
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent / "servers_config.json"
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Optional[Dict[str, Dict[str, Any]]] = None
    
    def load_config(self) -> Dict[str, Dict[str, Any]]:
        """加载MCP服务器配置"""
        if self._config is not None:
            return self._config
        
        config = {}
        
        # 从JSON文件加载
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                config.update(file_config)
                logger.info("从配置文件加载MCP服务: %s", list(config.keys()))

        # 替换占位符
        amap_key = settings.amap_api_key or os.environ.get("AMAP_API_KEY", "")
        for cfg in config.values():
            url = cfg.get("url", "")
            if "AMAP_API_KEY" in url and amap_key:
                cfg["url"] = url.replace("AMAP_API_KEY", amap_key)
        
        # 如果没有配置文件，使用默认配置（高德地图）
        if not config:
            logger.warning("未找到MCP配置文件，使用默认配置")
            amap_key = settings.amap_api_key or os.environ.get("AMAP_API_KEY", "")
            if not amap_key:
                logger.error("未设置 AMAP_API_KEY")
            config = {
                "amap": {
                    "url": f"{settings.amap_mcp_url}?key={amap_key}",
                    "transport": "sse",
                }
            }
        
        self._config = config
        return config
    
    def get_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """获取单个服务器配置"""
        config = self.load_config()
        return config.get(server_name)
    
    def list_servers(self) -> List[str]:
        """列出所有服务器"""
        return list(self.load_config().keys())
    
    def reload(self) -> None:
        """重新加载配置"""
        self._config = None
        self.load_config()


if __name__ == "__main__":
    print("=" * 50)
    print("MCP注册中心测试")
    print("=" * 50)
    
    registry = MCPServerRegistry()
    config = registry.load_config()
    
    print(f"\n已注册服务: {registry.list_servers()}")
    
    for name, cfg in config.items():
        print(f"\n服务: {name}")
        for key, value in cfg.items():
            # 隐藏API Key
            if "key" in key.lower() and value:
                value = value[:10] + "..." if len(value) > 10 else "***"
            print(f"  {key}: {value}")
    
    print("\n[OK] 配置加载成功!")
    print("=" * 50)
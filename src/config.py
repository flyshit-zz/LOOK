# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 使用默认值，BaseSettings 会自动从环境变量或 .env 文件加载
    deepseek_api_key: str = ""          # 环境变量名默认就是字段名（大写也可自动匹配）
    deepseek_model: str = "deepseek-chat"

    amap_api_key: str = ""
    amap_mcp_url: str = "https://mcp.amap.com/sse"

    # ── 记忆模块配置 ───────────────────────────────────────────────
    memory_db_path: str = "memory.db"
    """长时记忆 SQLite 数据库路径"""
    memory_chroma_dir: str = "chroma_memory_db"
    """长时记忆 ChromaDB 向量存储目录"""
    memory_token_budget: int = 6000
    """上下文组装 token 预算"""
    memory_embedding_model: str = "text-embedding-v3"
    """ChromaDB embedding 模型（DashScope TextEmbedding 模型名）"""
    dashscope_api_key: str = ""
    """阿里云 DashScope API Key（默认从 DASHSCOPE_API_KEY 环境变量读取）"""

    # Pydantic V2 配置
    model_config = SettingsConfigDict(
        env_file=".env",                # 指定 .env 文件路径（默认在当前目录）
        env_file_encoding="utf-8",
        extra="ignore",                 # 忽略未定义的额外字段，防止报错
    )

settings = Settings()

if __name__ == "__main__":
    print("=" * 50)
    print("配置测试")
    print("=" * 50)
    print(f"DeepSeek模型: {settings.deepseek_model}")
    print(f"DeepSeek API Key: {'已设置' if settings.deepseek_api_key else '❌ 未设置'}")
    print(f"高德API Key: {'已设置' if settings.amap_api_key else '❌ 未设置'}")
    print(f"高德MCP URL: {settings.amap_mcp_url}")
    print("=" * 50)
"""
配置加载模块 (Configuration Handler)
-------------------------------------
负责从 YAML 配置文件中加载各类配置（Prompts），
并将配置缓存为模块级全局变量供其他模块直接引用。

配置文件格式要求: yaml
yaml 语法特点: k: v（键值对）
"""
# 确保项目根目录在 sys.path 中（无论从哪里运行此文件都能找到 utils 等包）
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入第三方库 yaml，用于解析 YAML 格式的配置文件
import yaml
# 从项目工具模块导入路径处理函数，用于获取相对于项目根目录的绝对路径
from src.utils.path_tool import get_abs_path


def load_prompts_config(
    config_path: str = get_abs_path("config/prompts.yml"),  # Prompt 模板配置文件路径
    encoding: str = "utf-8"                                  # 文件编码，默认使用 utf-8
):
    """加载 Prompt 模板相关配置"""
    # 以只读模式、指定编码打开配置文件
    with open(config_path, "r", encoding=encoding) as f:
        # 使用 yaml.FullLoader 解析并返回完整配置字典
        return yaml.load(f, Loader=yaml.FullLoader)


# ---- 模块级单例：启动时即加载配置，避免每次调用都重新读取文件 ----

prompts_conf = load_prompts_config()   # 加载 Prompt 模板配置到全局变量  

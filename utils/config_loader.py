import yaml
import os

def load_config(config_path="config.yaml"):
    """加载 YAML 配置文件"""
    # 获取当前文件所在目录的绝对路径，以便正确找到 config.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    full_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(full_path):
        # 兜底：如果项目根目录下没找到，尝试在当前运行目录下找
        full_path = config_path
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Config file not found: {full_path}")
            
    with open(full_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 全局配置对象
config = load_config()

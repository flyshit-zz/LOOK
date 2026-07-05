"""
为整个工程提供绝对路径
"""
import os


def get_project_root():
    """
    获得工程的根目录
    """
    current_file=os.path.abspath(__file__)
    current_dir=os.path.dirname(current_file)
    project_root=os.path.dirname(current_dir)
    return  project_root

def get_abs_path(relative_path:str)->str:
    """
    传递相对路径，获得绝对路径
    param：relative_path 相对路径
    return 绝对路径
    """
    project_root=get_project_root()
    #将相对路径与根路径整合为绝对路径
    return os.path.join(project_root,relative_path)
    
if __name__=="__main__":
    print(get_abs_path("config/config"))
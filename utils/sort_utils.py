import pandas as pd
import os
import argparse

def sort_csv_stably(file_path):
    """
    对 CSV 文件按 ID 进行稳定排序
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件 {file_path} 不存在")
        return False
    
    try:
        print(f"正在读取文件: {file_path} ...")
        # 读取 CSV，确保 ID 列被正确解析
        df = pd.read_csv(file_path)
        
        if 'id' not in df.columns:
            print(f"❌ 错误: 文件中未找到 'id' 列")
            return False
            
        print(f"正在进行稳定排序 (Stable Sort by ID)...")
        # 使用 stable 排序算法确保同一 ID 的多条记录保持原始相对顺序
        df_sorted = df.sort_values(by="id", kind="stable")
        
        # 保存回原文件，使用 utf-8 无 BOM 编码
        df_sorted.to_csv(file_path, index=False, encoding='utf-8')
        print(f"✅ 排序完成并已保存至: {file_path}")
        return True
    except Exception as e:
        print(f"❌ 排序过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对评论分类结果 CSV 进行稳定排序")
    parser.add_argument("--file", type=str, default="data/Result/Result_2.csv", help="要排序的 CSV 文件路径")
    
    args = parser.parse_args()
    sort_csv_stably(args.file)

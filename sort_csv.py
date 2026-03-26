from utils.sort_utils import sort_csv_stably
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对评论分类结果 CSV 进行稳定排序")
    parser.add_argument("--file", type=str, default="data/Result/Result_2.csv", help="要排序的 CSV 文件路径")
    
    args = parser.parse_args()
    sort_csv_stably(args.file)

import pandas as pd
import os

def check_missing(reviews_csv, results_csv):
    if not os.path.exists(reviews_csv):
        print(f"Error: {reviews_csv} not found.")
        return
    if not os.path.exists(results_csv):
        print(f"Error: {results_csv} not found.")
        return

    # 1. 加载原始测试评论 ID
    df_reviews = pd.read_csv(reviews_csv)
    all_ids = set(df_reviews['id'].astype(int))
    total_count = len(all_ids)

    # 2. 加载已跑出的结果 ID
    # 由于结果文件可能没有 header，我们尝试自动识别
    try:
        # 先尝试按有 header 读取
        df_results = pd.read_csv(results_csv)
        if 'id' in df_results.columns:
            processed_ids = set(df_results['id'].astype(int))
        else:
            # 如果没有 'id' 列，尝试按无 header 读取第一列
            df_results = pd.read_csv(results_csv, header=None)
            processed_ids = set(df_results[0].astype(int))
    except Exception as e:
        print(f"Error reading results: {e}")
        return

    processed_count = len(processed_ids)

    # 3. 计算遗漏
    missing_ids = all_ids - processed_ids
    
    print(f"--- Data Integrity Check ---")
    print(f"Total reviews in test set: {total_count}")
    print(f"Unique IDs in result file: {processed_count}")
    
    if not missing_ids:
        print("✅ No missing reviews! All IDs are present in the result file.")
    else:
        print(f"❌ Found {len(missing_ids)} missing reviews.")
        print(f"Completion rate: {(processed_count/total_count)*100:.2f}%")
        print(f"Missing IDs (first 50): {sorted(list(missing_ids))[:50]}")

if __name__ == "__main__":
    test_reviews = "data/TEST/Test_reviews.csv"
    result_path = "data/result.csv" 
    if not os.path.exists(result_path):
        result_path = "data/Result.csv"
        
    check_missing(test_reviews, result_path)

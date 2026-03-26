import pandas as pd
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent import run_agent
from utils.config_loader import config
import json
import time

# 线程锁，确保 CSV 写入安全
csv_lock = threading.Lock()

def process_single_review(review_id, review_text, output_csv):
    """
    单个评论的处理逻辑
    """
    current_batch = []
    try:
        # 调用 agent.py 中的 run_agent
        # 注意：在并发模式下，建议关闭 agent.py 内部的冗余打印以提升性能
        final_state = run_agent(review_text)
        results = final_state.get("results", [])
        
        if not results:
            current_batch.append({
                "id": review_id,
                "AspectTerms": "_",
                "OpinionTerms": "_",
                "Categories": "其他",
                "Polarities": "中性"
            })
        else:
            for res in results:
                current_batch.append({
                    "id": review_id,
                    "AspectTerms": res.get("AspectTerms", "_"),
                    "OpinionTerms": res.get("OpinionTerms", "_"),
                    "Categories": res.get("Categories", "_"),
                    "Polarities": res.get("Polarities", "_")
                })
        
        # 实时写入文件（线程安全）
        if current_batch:
            df_current = pd.DataFrame(current_batch)
            with csv_lock:
                # 只有文件不存在或者文件为空时，才写入表头
                should_write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
                df_current.to_csv(
                    output_csv, 
                    mode='a', 
                    header=should_write_header, 
                    index=False, 
                    encoding='utf-8'
                )
                print(f"✅ ID {review_id} results saved to {output_csv}")
        return True
    except Exception as e:
        print(f"❌ Error ID {review_id}: {e}")
        return False

def batch_test_parallel(input_csv, output_csv, max_workers=10):
    """
    并行批量测试
    """
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    # 1. 加载测试数据
    df_test = pd.read_csv(input_csv)
    
    # 2. 自动断点续跑
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            # 采用逐行读取方式提取 ID，以应对可能的脏数据
            processed_ids = set()
            with open(output_csv, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if parts[0].isdigit():
                        processed_ids.add(int(parts[0]))
            
            df_to_process = df_test[~df_test["id"].astype(int).isin(processed_ids)]
            print(f"Detected {len(processed_ids)} already processed unique IDs. Skipping them.")
        except Exception as e:
            print(f"Warning: Could not extract processed IDs: {e}. Starting fresh.")
            df_to_process = df_test
    else:
        df_to_process = df_test

    if df_to_process.empty:
        print("All records have been processed. Nothing to do.")
        return

    print(f"Starting parallel processing with {max_workers} workers. Total to process: {len(df_to_process)}")
    start_time = time.time()

    # 3. 使用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 list 提交任务以保持提交顺序，虽然执行顺序不固定
        future_to_id = {executor.submit(process_single_review, row['id'], row['Reviews'], output_csv): row['id'] for _, row in df_to_process.iterrows()}
        
        completed_count = 0
        for future in as_completed(future_to_id):
            rid = future_to_id[future]
            completed_count += 1
            
            # 每处理完一个，可以稍微 sleep 一下，缓解 API 并发压力
            time.sleep(0.1) 

            if completed_count % 10 == 0 or completed_count == len(df_to_process):
                elapsed = time.time() - start_time
                speed = completed_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed_count}/{len(df_to_process)} (Speed: {speed:.2f} reviews/s)")

    end_time = time.time()
    print(f"\nSuccessfully finished. Total time: {end_time - start_time:.2f}s")
    
    from utils.sort_utils import sort_csv_stably
    print("Finalizing results: sorting by ID (stable sort)...")
    if sort_csv_stably(output_csv):
        print("✅ Results sorted stably and saved.")
    else:
        print(f"Warning: Could not sort final CSV")

if __name__ == "__main__":
    # 配置参数
    batch_conf = config.get('batch', {})
    INPUT_PATH = batch_conf.get('input_path', "data/TEST/Test_reviews.csv")
    OUTPUT_PATH = batch_conf.get('output_path', "data/Result/Result_3.csv")
    CONCURRENCY = batch_conf.get('concurrency', 30)
    
    batch_test_parallel(INPUT_PATH, OUTPUT_PATH, max_workers=CONCURRENCY)

from rag.ragservice import RAGService
from utils.tools import MySQLService
import config
import os

def main():
    # 1. 初始化 RAG
    print("正在初始化 RAG 向量数据库...")
    rag = RAGService(persist_directory="./chroma_db")
    rag.initialize_from_csv("data/TRAIN/Train_reviews.csv")
    
    # 2. 初始化 MySQL
    print("正在初始化 MySQL 数据库...")
    mysql_service = MySQLService(password = config.get('mysql', {}).get('password'))
    try:
        mysql_service.initialize_db()
        mysql_service.load_labels_from_csv("data/TRAIN/Train_labels.csv")
    except Exception as e:
        print(f"MySQL 初始化失败: {e}")
        print("请确保 MySQL 服务已启动且密码正确。")

if __name__ == "__main__":
    main()

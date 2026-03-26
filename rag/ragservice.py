import os
import pandas as pd
import threading
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

class RAGService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self, persist_directory="./chroma_db"):
        # 确保只初始化一次
        if hasattr(self, 'initialized') and self.initialized:
            return
        
        self.persist_directory = persist_directory
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v4")
        self.vector_db = None
        self.initialized = True
        self.load_db() # 预加载以避免并发冲突

    def initialize_from_csv(self, csv_path):
        """从 CSV 文件初始化向量数据库"""
        if not os.path.exists(csv_path):
            print(f"CSV file {csv_path} not found.")
            return

        df = pd.read_csv(csv_path)
        documents = []
        for index, row in df.iterrows():
            # 将 review 内容存入向量库，metadata 存储 id 以便关联 MySQL
            doc = Document(
                page_content=str(row['Reviews']),
                metadata={"id": int(row['id'])}
            )
            documents.append(doc)

        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Initialized RAG with {len(documents)} documents.")

    def load_db(self):
        """加载已存在的向量数据库"""
        if os.path.exists(self.persist_directory):
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Loaded existing RAG database.")
        else:
            print("No existing RAG database found. Please initialize first.")

    def search(self, query, k=3):
        """检索最相似的评论"""
        if self.vector_db is None:
            self.load_db()
        
        if self.vector_db:
            results = self.vector_db.similarity_search_with_score(query, k=k)
            return [{"id": res[0].metadata["id"], "content": res[0].page_content, "score": res[1]} for res in results]
        return []

if __name__ == "__main__":
    # 示例用法
    rag = RAGService()
    # rag.initialize_from_csv("data/TRAIN/Train_reviews.csv")
    print(rag.search("十年前用过这个品牌的隔离霜，没想到现在又用上了，效果不错"))

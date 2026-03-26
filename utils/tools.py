import pymysql
import pandas as pd
from sqlalchemy import create_engine
from .config_loader import config

class MySQLService:
    def __init__(self, user=None, password=None, host=None, database=None):
        mysql_conf = config.get('mysql', {})
        self.user = user or mysql_conf.get('user', 'root')
        self.password = password or mysql_conf.get('password', '')
        self.host = host or mysql_conf.get('host', 'localhost')
        self.database = database or mysql_conf.get('database', 'review_db')
        self.engine = create_engine(f"mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}")

    def initialize_db(self):
        """创建数据库（如果不存在）"""
        conn = pymysql.connect(host=self.host, user=self.user, password=self.password)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        conn.close()

    def load_labels_from_csv(self, csv_path, table_name='labels'):
        """将 CSV 标签数据加载到 MySQL (过滤冗余字段)"""
        df = pd.read_csv(csv_path)
        
        # 过滤掉 A_start, A_end, O_start, O_end 字段
        cols_to_drop = ['A_start', 'A_end', 'O_start', 'O_end']
        # 检查列是否存在再删除，防止报错
        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols_to_drop)
        
        # 将数据写入 MySQL
        df.to_sql(table_name, con=self.engine, if_exists='replace', index=False)
        print(f"Loaded {len(df)} labels into MySQL table '{table_name}' (Filtered: {existing_cols_to_drop}).")

    def get_labels_by_id(self, review_id, table_name='labels'):
        """根据 ID 查询标签"""
        query = f"SELECT * FROM {table_name} WHERE id = {review_id}"
        df = pd.read_sql(query, con=self.engine)
        return df.to_dict(orient='records')

if __name__ == "__main__":
    # 示例用法
    mysql_service = MySQLService(password='Mqn12138?')
    mysql_service.initialize_db()
    mysql_service.load_labels_from_csv("data/TRAIN/Train_labels.csv")
    print("Labels loaded successfully.")

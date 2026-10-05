import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT")),
        charset="utf8mb4"
    )
    return conn

def init_table():
    conn = get_db_connection()
    cur = conn.cursor()

    # 用户表
    cur.execute("""
    CREATE TABLE IF NOT EXISTS `user` (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) NOT NULL UNIQUE,
        password VARCHAR(100) NOT NULL,
        role ENUM('student','admin') DEFAULT 'student',
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 作文提交记录表
    cur.execute("""
    CREATE TABLE IF NOT EXISTS writing_submit (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        content TEXT,
        score TINYINT,
        comment TEXT,
        revise_content TEXT,
        submit_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 翻译提交记录表
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trans_submit (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        origin_text TEXT,
        user_trans TEXT,
        score TINYINT,
        suggest TEXT,
        submit_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 错题记录表
    cur.execute("""
    CREATE TABLE IF NOT EXISTS error_record (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        question_id INT,
        error_type ENUM('vocab','grammar','method','careless'),
        source_type ENUM('writing','trans','manual') DEFAULT NULL,
        source_id INT DEFAULT NULL,
        first_wrong DATETIME DEFAULT CURRENT_TIMESTAMP,
        review_d1 DATETIME NULL,
        review_d3 DATETIME NULL,
        review_d7 DATETIME NULL,
        review_d15 DATETIME NULL,
        finished TINYINT DEFAULT 0
    )
    """)

    # 兼容旧表：自动添加 source_type / source_id 列（如果不存在）
    try:
        cur.execute("ALTER TABLE error_record ADD COLUMN source_type ENUM('writing','trans','manual') DEFAULT NULL")
    except Exception:
        pass  # 列已存在
    try:
        cur.execute("ALTER TABLE error_record ADD COLUMN source_id INT DEFAULT NULL")
    except Exception:
        pass  # 列已存在

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    init_table()
    print("数据表初始化完成")
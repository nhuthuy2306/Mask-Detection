"""
Migration script: Thêm 3 cột đo độ trễ vào bảng detection_logs trên Supabase.
Chạy 1 lần duy nhất: python migrate_latency.py
"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Chưa cấu hình DATABASE_URL trong .env")

COLUMNS_TO_ADD = [
    ("inference_time_ms", "DOUBLE PRECISION"),  # a) thời gian suy luận model (ms)
    ("roundtrip_time_ms", "DOUBLE PRECISION"),  # b) roundtrip local <-> Cloud (ms)
    ("total_latency_ms",  "DOUBLE PRECISION"),  # c) độ trễ toàn trình (ms)
]

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

for col_name, col_type in COLUMNS_TO_ADD:
    sql = f"ALTER TABLE detection_logs ADD COLUMN IF NOT EXISTS {col_name} {col_type};"
    try:
        cur.execute(sql)
        print(f"[OK] Đã thêm cột: {col_name} ({col_type})")
    except Exception as e:
        print(f"[ERR] {col_name}: {e}")

conn.commit()
cur.close()
conn.close()
print("\nMigration hoàn tất!")


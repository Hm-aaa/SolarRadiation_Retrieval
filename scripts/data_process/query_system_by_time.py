# --代码功能-------------------------------
# --查询指定时间范围内的环测仪业务库内可用的环测仪id

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
from dateutil.relativedelta import relativedelta


# 查询时间范围
start_time = datetime(2023, 1, 1, 9, 0, 0)
end_time = datetime(2026, 10, 16, 9, 30, 0)

# CSV 输出目录
output_dir = Path("/home/jinhuaiming/SolarRadiation_Retrieval/data/hcy_system_id")

# 数据库连接
conn = psycopg2.connect(
    host="118.178.121.69",
    port=15432,
    dbname="secp",
    user="sebu_algorithm_readonly",
    password="sebu_algorithm_readonly",  # 或直接填写密码
    options="-c statement_timeout=30000",
)

conn.set_session(readonly=True, autocommit=True)
cursor = conn.cursor()

sql = """
    SELECT DISTINCT system_id
    FROM meta.device_env_5min_t1_basic_data
    WHERE date_time = %s
      AND BTRIM(data_body ->> 'irradiance') <> ''
    ORDER BY system_id
"""

current = start_time
query_count = 0
saved_file_count = 0
saved_row_count = 0

try:
    while current <= end_time:
        query_count += 1

        # 如果文件已经存在，说明之前执行过，直接跳过
        # ---- 路径逻辑：根目录 / 年 / 月 / 日 / 文件名 ----
        day_dir = output_dir / f"{current:%Y}" / f"{current:%m}" / f"{current:%d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"hcy_id_{current:%Y%m%d_%H%M%S}.csv"
        output_path = day_dir / file_name

        if output_path.exists():
            print(f'{current}:文件存在')
            current += timedelta(minutes=5)
            continue
        
        cursor.execute(sql, (current,))
        rows = cursor.fetchall()
        # 没有结果，不创建文件
        if rows:
            with output_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["system_id"])
                writer.writerows(rows)

            saved_file_count += 1
            saved_row_count += len(rows)

            print(
                f"{current:%Y-%m-%d %H:%M:%S}，"
                f"找到 {len(rows)} 个 system_id，已保存 {file_name}"
            )
        
        if query_count % 1000 == 0:
            print(
                f"进度：{current:%Y-%m-%d %H:%M:%S}，"
                f"已查询 {query_count} 次，"
                f"已生成 {saved_file_count} 个文件"
            )

        current += timedelta(minutes=5)
finally:
    cursor.close()
    conn.close()

print("查询完成")
print("查询次数：", query_count)
print("生成文件数：", saved_file_count)
print("导出记录数：", saved_row_count)
print("输出目录：", output_dir)
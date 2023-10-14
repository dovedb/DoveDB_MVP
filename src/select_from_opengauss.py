import json
import psycopg2
from psycopg2 import sql

# 数据表定义
# CREATE TABLE cars (
#     boxid SERIAL PRIMARY KEY,  -- auto incrementing integer
#     frameid INT NOT NULL,
#     timestamp INT NOT NULL,
#     left INT NOT NULL,
#     right INT NOT NULL,
#     top INT NOT NULL,
#     bottom INT NOT NULL,
#     label VARCHAR(255) NOT NULL,  -- assuming label as a variable-length string
#     confidence FLOAT NOT NULL,
#     track_id INT NOT NULL
# );

db_params = {
    'dbname':'dovedb_offline',
    'user':'',
    'password':'',
    'host':'localhost',
    'port':'5432',
}

# SQL query
sql_query = """
SELECT COUNT(DISTINCT track_id) 
FROM cars 
WHERE timestamp BETWEEN 60 AND 120 
AND confidence > 0.7;
"""
vmql_query = """
SELECT COUNT(DISTINCT track_id) 
FROM cars 
WHERE timestamp BETWEEN 60 AND 120 
AND confidence > 0.7;
"""

# Connect to the database
with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        result = cursor.fetchone()
        print(f"Number of distinct cars: {result[0]}")


# SQL query to count cars per frame, ordered by count descending
sql_query = """
SELECT frameid, COUNT(track_id) as car_count 
FROM cars 
GROUP BY frameid 
ORDER BY car_count DESC;
"""
vmql_query = """
SELECT frameid, COUNT(track_id) as car_count 
FROM cars 
GROUP BY frameid 
ORDER BY car_count DESC
LIMIT 5;
"""

def filter_frames(cursor, k, min_interval):
    selected_frames = []
    previous_timestamp = None

    while len(selected_frames) < k:
        frameid, _ = cursor.fetchone()
        if previous_timestamp is None or (frameid - previous_timestamp) >= min_interval:
            selected_frames.append(frameid)
            previous_timestamp = frameid

    return selected_frames

# Connect to the database
with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        top_frames = filter_frames(cursor, k=5, min_interval=3)
        print(f"Top {len(top_frames)} frames with most cars: {', '.join(map(str, top_frames))}")


# SQL query
sql_query = """
SELECT 
    MAX(timestamp) - MIN(timestamp) as stay_duration
FROM 
    cars 
WHERE 
    track_id = 20;
"""
vmql_query = """
SELECT 
    MAX(timestamp) - MIN(timestamp) as stay_duration
FROM 
    cars 
WHERE 
    track_id = 20;
"""

# Connect to the database
with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        result = cursor.fetchone()
        if result[0] is not None:
            print(f"Stay duration of car with track_id=20: {result[0]} seconds")
        else:
            print("No data available for track_id=20")

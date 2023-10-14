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


with open('parsed_data.json', 'r', encoding='utf8') as f:
    data = json.load(f)


dbname = 'dovedb_offline'
user = ''
password = ''
host = 'localhost'
port = '5432'
conn = psycopg2.connect(database=dbname, user=user, password=password, host=host, port=port)
cursor = conn.cursor()


fps = 30
for frameid, frame_data in enumerate(data):
    timestamp = frameid // fps
    for object_data in frame_data:
        left = object_data['left']
        right = object_data['right']
        top = object_data['top']
        bottom = object_data['bottom']
        label = object_data['class']
        confidence = object_data['score']
        track_id = object_data['track_id']
        # Insert the data row into the PostgreSQL table
        insert_query = sql.SQL(
            "INSERT INTO cars (frameid, timestamp, \"left\", \"right\", \"top\", \"bottom\", label, confidence, track_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"
        )

        cursor.execute(insert_query, (frameid, timestamp, left, right, top, bottom, label, confidence, track_id))

# Commit all the inserts and close the database connection
conn.commit()
cursor.close()
conn.close()
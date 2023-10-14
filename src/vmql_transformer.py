from lark import Lark, Transformer
import psycopg2
from psycopg2 import sql
from config import db_params

# Define the grammar
grammar = """
    start: stmt ";"
    
    stmt: create_stmt | select_stmt

    create_stmt: "CREATE" object_type SIGNED_INT? CNAME "ON" dataset 
               | "CREATE" object_type CNAME "USING" CNAME "ON" dataset where_clause?

    object_type: "YOLO" | "MONITOR_EVENT"
    
    dataset: CNAME "(" CNAME ("," CNAME)* ")" 
       | "DATA_SOURCE" CNAME
       | CNAME   // This accounts for table names like CARS
       | "(" select_stmt ")"

    where_clause: "WHERE" condition
    condition: CNAME COMPARATOR value
             | condition LOGICAL_OPERATOR condition

    LOGICAL_OPERATOR: "AND" | "OR"

    select_stmt: "SELECT" fields "FROM" dataset group_by_clause? where_clause? order_by_clause? limit_clause?

    group_by_clause: "GROUP BY" CNAME

    fields: field ("," field)*
    
    field: CNAME
         | function_call alias?
         | function_call ARITHMETICAL_OP function_call alias?
    
    ARITHMETICAL_OP: "+" | "-" | "*" | "/"

    alias: "AS" CNAME

    aggregation: DISTINCT? CNAME
               | "*"

    DISTINCT: "DISTINCT"

    aggregation_op: "MAX" | "MIN"

    function_call: CNAME "(" aggregation ")"
    
    order_by_clause: "ORDER BY" CNAME order? 

    limit_clause: "LIMIT" SIGNED_INT

    order: asc | desc

    asc: "ASC"
    desc: "DESC"

    COMPARATOR: "=" | ">" | "<" | ">=" | "<="

    value: SIGNED_INT | FLOAT | string | "NOW()" "-" SIGNED_INT

    FLOAT: /-?\d+\.\d+/
    
    string: ESCAPED_STRING

    %import common.ESCAPED_STRING

    %import common.CNAME
    %import common.SIGNED_INT
    %import common.WS

    %ignore WS
"""


class SQLTransformer(Transformer):
    # Handle the entire statement
    def start(self, items):
        return items[0]
    
    # Handle both create and select statements
    def stmt(self, items):
        return items[0] + ';'
    
    def create_stmt(self, items):
        return ' '.join(map(str, items))

    def object_type(self, items):
        return items[0]

    def dataset(self, items):
        return 'FROM ' + ''.join(map(str, items))
    
    def where_clause(self, items):
        return "WHERE " + str(items[0])

    def condition(self, items):
        return ' '.join(map(str, items))

    def select_stmt(self, items):
        return 'SELECT ' + ' '.join(filter(None, items))

    def group_by_clause(self, items):
        return 'GROUP BY ' + str(items[0])

    def fields(self, items):
        return ', '.join(items)

    def field(self, items):
        if len(items) == 1:
            return str(items[0])
        return ' '.join(items[:-1]) + ' AS ' + items[-1]
    
    def ARITHMETICAL_OP(self, items):
        return items
    
    def alias(self, items):
        return items[0]
    
    def LOGICAL_OPERATOR(self, items):
        return items
    
    def DISTINCT(self, items):
        return items

    def aggregation(self, items):
        return ' '.join(items)

    def function_call(self, items):
        return f"{items[0]}({items[1]})"

    def order_by_clause(self, items):
        # If the order (ASC/DESC) is not provided, it will default to an empty string
        order_direction = items[1] if len(items) > 1 else ''
        return 'ORDER BY ' + items[0] + ' ' + order_direction

    def order(self, items):
        return items[0].data.upper()

    def limit_clause(self, items):
        # We're capturing the LIMIT value, but not using it in the SQL statement
        DEFAULT_INTERVAL = 3
        self.limit_value = int(items[0])
        self.min_interval = DEFAULT_INTERVAL
        return None

    def cname(self, items):
        return items[0]

    def COMPARATOR(self, items):
        return items[0]

    def value(self, items):
        return ''.join(map(str, items))

    # Since we're not using the LIMIT in the transformed SQL, we don't need a method for it here

    # Convert terminal tokens to string
    def FLOAT(self, token):
        return token

    def string(self, items):
        return items[0]

    def CNAME(self, token):
        return token

    def SIGNED_INT(self, token):
        return token


# Create the Lark parser
parser = Lark(grammar, start='start')

# SQL statements to parse
# sql_statements = [
#     "CREATE YOLO red_car_detector ON traingin_set(img, red_car_labels);",
#     "CREATE MONITOR_EVENT red_car_event USING red_car_detector ON DATA_SOURCE online_cameras WHERE cam_id < 30;",
#     "CREATE MONITOR_EVENT mot_for_cars USING mot ON DATA_SOURCE online_cameras;",
#     "SELECT COUNT(DISTINCT car_id) FROM (SELECT car_id FROM mot(online_cameras) WHERE cam_id=102 and timestamp > now()-300);",
#     "SELECT frameid, timestamp, COUNT(track_id) as car_count FROM mot(online_cameras) WHERE cam_id=0 and timestamp > now()-300 ORDER BY car_count DESC LIMIT 10;",
#     "SELECT frameid, COUNT(track_id) as car_count FROM cars GROUP BY frameid ORDER BY car_count DESC LIMIT 5;",
#     "SELECT COUNT(DISTINCT track_id) FROM cars WHERE timestamp BETWEEN 60 AND 120 AND confidence > 0.7;",
#     "SELECT MAX(timestamp) - MIN(timestamp) as stay_duration FROM cars WHERE track_id = 20;"
# ]


def query_opengauss(sql, post_process, **kargs):
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = post_process(cursor, **kargs)
    return result


def filter_frames(cursor, k, min_interval):
    selected_frames = []
    previous_timestamp = None

    while len(selected_frames) < k:
        frameid, _ = cursor.fetchone()
        if previous_timestamp is None or (frameid - previous_timestamp) >= min_interval:
            selected_frames.append(frameid)
            previous_timestamp = frameid
    selected_frames = list(map(str, selected_frames))
    return 'Selected frames: ' + ', '.join(selected_frames)


def fetch_one(cursor):
    result = cursor.fetchone()
    return str(result[0])


def process_vmql(vmql):
    # vmql = 'SELECT frameid, COUNT(track_id) as car_count FROM cars GROUP BY frameid ORDER BY car_count DESC LIMIT 5;'
    # vmql = 'SELECT COUNT(DISTINCT track_id) FROM cars WHERE timestamp < 120 AND confidence > 0.7;'
    # vmql = 'SELECT MAX(timestamp) - MIN(timestamp) as stay_duration FROM cars WHERE track_id = 20;'
    tree = parser.parse(vmql.upper())
    transformer = SQLTransformer()
    # Transform the tree
    transformed_sql = transformer.transform(tree)
    print(transformed_sql)

    if hasattr(transformer, 'limit_value'):
        result = query_opengauss(transformed_sql, 
                                 filter_frames, 
                                 k=transformer.limit_value, 
                                 min_interval=transformer.min_interval)
    else:
        result = query_opengauss(transformed_sql, fetch_one)
    print(result)
    return result


if __name__ == '__main__':
    process_vmql('SELECT frameid, COUNT(track_id) as car_count FROM cars GROUP BY frameid ORDER BY car_count DESC LIMIT 5;')

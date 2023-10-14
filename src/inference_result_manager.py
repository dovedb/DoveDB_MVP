import psycopg2


class InferenceResultManager(object):

    def __init__(self, engine_setttings):
        self.conn = psycopg2.connect(database=engine_setttings['database'],
                        host=engine_setttings['host'],
                        user=engine_setttings['user'],
                        password=engine_setttings['password'],
                        port=engine_setttings['port'])
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cars")
        print(cursor.fetchall())

    def insert(self, event_id, frame_id, result):
        """Insert inference result.

        Args:
            event_id (int): Event ID.
            frame_id (int)： Frame ID.
            result (dict)： Inference result. A key-value dict that
                contains the inference result. Defined by the event.

        Returns:
            bool: True if success, otherwise False.

        Raises:
            ValueError: If event_id or frame_id is invalid.
        """
        

    def query(self):
        """"Query inference result.

        Args:
            query (Lark.Tree): Query. A SQL-Like string that defines the query.

        Returns:
            List: A list of inference result.
        """



if __name__ == '__main__':
    setting = {
        'database': 'dovedb_offline',
        'host': 'localhost',
        'user': '',
        'password': '',
        'port': '5432'
    }
    manager = InferenceResultManager(setting)
    manager.insert(event_id=0, frame_id=0, result={'car_id': '0', 'x': 0.34, 'y': 0.56, 'w': 0.45, 'h': 0.53})
    manager.insert(event_id=0, frame_id=0, result={'car_id': '1', 'x': 0.05, 'y': 0.13, 'w': 0.46, 'h': 0.10})
    manager.insert(event_id=0, frame_id=0, result={'car_id': '2', 'x': 0.32, 'y': 0.41, 'w': 0.32, 'h': 0.59})
    manager.insert(event_id=0, frame_id=0, result={'car_id': '3', 'x': 0.12, 'y': 0.50, 'w': 0.11, 'h': 0.04})
    manager.insert(event_id=0, frame_id=1, result={'car_id': '0', 'x': 0.67, 'y': 0.59, 'w': 0.53, 'h': 0.40})
    manager.insert(event_id=0, frame_id=1, result={'car_id': '1', 'x': 0.54, 'y': 0.54, 'w': 0.29, 'h': 0.32})

    from server.executor import QueryExecutor

    executor = QueryExecutor()
    command = 'SELECT COUNT(car_id) FROM (SELECT car_id FROM mot(online_cameras) WHERE cam_id=102 AND timestamp > now(a) - 300);'
    # command = 'SELECT COUNT(DISTINCT car_id) FROM (SELECT car_id FROM mot(online_cameras) WHERE cam_id=102 and timestamp > now()-300);'
    tree = executor.parse(command)

    print(tree)
    print(type(tree))

    # manager.query('')


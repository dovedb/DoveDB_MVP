from lark import Lark, Transformer, v_args


vmql_grammar = """
    start: command
    command: create ";" | select ";"
    create: "CREATE" model CNAME "ON" CNAME "(" CNAME "," CNAME ")"
    select: "SELECT" CNAME "FROM" "(" CNAME "(" CNAME ")" "WHERE" condition ")"
    model: CNAME
    condition: CNAME _compare CNAME
    _compare: "=" | ">" | "<" | ">=" | "<=" | "!="

    %import common.CNAME
    %import common.WS
    %ignore WS
"""


class CommandTransformer(Transformer):
    def create(self, items):
        model, name, dataset, img, labels = items
        return {"type": "create", "model": str(model), "name": str(name), "dataset": str(dataset), "img": str(img), "labels": str(labels)}

    def select(self, items):
        obj, name, source, condition = items
        return {"type": "select", "object": str(obj), "name": str(name), "source": str(source), "condition": str(condition)}


class QueryExecutor(object):
    """DoveDB server."""

    def __init__(self):
        pass

    def parse(self, query):
        """Parse query.
        
        Args:
            query (str): Query to be parsed. The language is VMQL, SQL-like.
        
        Returns:
            dict: Parsed query. The format is as follows:
                {
                    'type': 'select',
                    'object': ['count(car)'],
                    'video': 'camera1',
                    'where': {
                        'id': 1
                    }
                }

        Raises:
            ValueError: If query is invalid.
        """
        return self.parse_command(command)

    def parse_command(self, command):
        parser = Lark(vmql_grammar, parser='lalr', transformer=CommandTransformer())
        tree = parser.parse(command)
        return tree

    def execute(self, query_parsed):
        """Execute query.
        
        Args:
            query_parsed (dict): Parsed query.
        
        Returns:
            list: Query result. The format is as follows:
                [
                    {
                        'count(car)': 20
                    }
                ]
        
        Raises:
            ValueError: If query is invalid.
        """
        pass


if __name__ == '__main__':
    executor = QueryExecutor()
    command = "CREATE YOLO red_car_detector ON training_set(img, red_car_labels);"
    print(executor.parse_command(command))


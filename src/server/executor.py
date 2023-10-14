from lark import Lark, Transformer, v_args


vmql_grammar = """
    start: command
    command: create ";" | select ";"
    create: "CREATE" model CNAME "ON" CNAME "(" CNAME "," CNAME ")"
    select: "SELECT" expr "FROM" target [where_clause] | "(" select ")"
    where_clause: "WHERE" condition
    target: select | CNAME | FUNCTION_CALL
    model: CNAME
    condition: compare_operation | condition logical_operator compare_operation
    logical_operator: "AND" | "OR"
    compare_operation: expr _compare expr
    _compare: "=" | ">" | "<" | ">=" | "<=" | "!="

    expr: term 
        | expr "+" term -> add
        | expr "-" term -> subtract
    
    term: factor
        | term "*" factor -> multiply
        | term "/" factor -> divide

    factor: NUMBER
           | "(" expr ")"
           | CNAME
           | FUNCTION_CALL

    FUNCTION_CALL: CNAME "(" [DISTINCT] (CNAME ("," CNAME)*)? ")"
    DISTINCT: "DISTINCT"

    %import common.NUMBER
    %import common.CNAME
    %import common.WS
    %ignore WS
"""

#     FUNCTION_CALL: CNAME "(" (expr ("," expr)*)? ")"

class CommandTransformer(Transformer):
    def create(self, items):
        model, name, dataset, img, labels = items
        return {"type": "create", "model": str(model), "name": str(name), "dataset": str(dataset), "img": str(img), "labels": str(labels)}

    def select(self, items):
        print(items)
        obj, source, condition = items
        return {"type": "select", "object": str(obj), "source": str(source), "condition": str(condition)}


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
        return self.parse_command(query)

    def parse_command(self, command):
        parser = Lark(vmql_grammar, parser='lalr')
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
    print(executor.parse(command))


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
        return {"type": "create", "model": model.children[0], "name": name, "dataset": dataset, "img": img, "labels": labels}

    def select(self, items):
        obj, name, source, condition = items
        return {"type": "select", "object": obj, "name": name, "source": source, "condition": condition}

def parse_command(command):
    parser = Lark(vmql_grammar, parser='lalr', transformer=CommandTransformer())
    result = parser.parse(command)
    return result

command = "CREATE YOLO red_car_detector ON training_set(img, red_car_labels);"
print(parse_command(command))



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
        pass

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

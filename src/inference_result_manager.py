

class InferenceResultManager(object):
    """Inference result manager."""

    def __init__(self):
        pass

    def insert_result(self, event_id, frame_id, result):
        """Insert inference result.

        Args:
            event_id (int): Event ID.
            frame_id (int): Frame ID.
            result (dict): Inference result. A key-value dict that 
                contains the inference result. Defined by the event.

        Returns:
            bool: True if success, otherwise False.

        Raises:
            ValueError: If event_id or frame_id is invalid.
        """

    def query_result(self, query):
        """Query inference result.
        
        Args:
            query (str): Query. A SQL-like string that defines the query.
        
        Returns:
            list: A list of inference result.
        """

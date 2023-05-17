

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
        
    def import_video_metadata(self,frame_data):

        query = """
        INSERT VERTEX Frame(video_id, frame_number)
        VALUES "{video_id}":("{video_id}", {frame_number})
        """
        for frame in frame_data:
            video_id = frame['video_id']
            frame_number = frame['frame_number']
            insert_query = query.format(video_id=video_id, frame_number=frame_number)
            print(insert_query) 
            reps = self.client.execute(insert_query)
            assert reps.is_succeeded(), reps.error_msg()
            
    def import_video_object(self,object_data):
        query = """
        INSERT VERTEX Object(object_id, class, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        VALUES "{object_id}":("{object_id}", "{class_name}", {bbox_xmin}, {bbox_ymin}, {bbox_xmax}, {bbox_ymax})
        """
        for obj in object_data:
            object_id = obj['object_id']
            class_name = obj['class']
            bbox_xmin,bbox_ymin = obj['bbox_xmin'],obj['bbox_ymin']
            bbox_xmax,bbox_ymax = obj['bbox_xmax'],obj['bbox_ymax']
            insert_query = query.format(object_id=object_id, class_name=class_name,bbox_xmin=bbox_xmin,
                                        bbox_ymin=bbox_ymin,bbox_xmax=bbox_xmax,bbox_ymax=bbox_ymax)
            print(insert_query)
            reps = self.client.execute(insert_query)
            assert reps.is_succeeded(), reps.error_msg()

    def insert_frame_correspondence_edges(self,frame_conatin_bbox_data):
        query = """
            INSERT EDGE frame_contains_object()
            VALUES "{video_id},{frame_number}"->"{object_id}"
        """
        for frame_conatin_bbox in frame_conatin_bbox_data:
            video_id = frame_conatin_bbox['video_id']
            frame_number = frame_conatin_bbox['frame_number']
            object_id = frame_conatin_bbox['object_id']
            insert_query = query.format(
                video_id=video_id, frame_number=frame_number
                ,object_id=object_id)
            
            print(insert_query)
            self.client.execute(insert_query)
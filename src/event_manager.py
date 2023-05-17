

class EventManager(object):
    """Event manager."""

    def __init__(self):
        pass

    def create_event(self, model_id, trigger, callback):
        """create a new event.
        
        Args:
            model_id (int): Model ID.
            trigger (str): Trigger.
            callback (str): Callback.
        
        Returns:
            int: Event ID.
        """

    def delete_event(self, event_id):
        """Delete an event.

        Args:
            event_id (int): Event ID.

        Returns:
            bool: True if success, otherwise False.

        Raises:
            ValueError: If event_id is invalid.
        """
    
    def mount(self, event_id, video_name):
        """Mount an event on a stream.
        
        Args:
            event_id (int): Event ID.
            video_name (str): Video name.
        
        Returns:
            bool: True if success, otherwise False.

        Raises:
            ValueError: If event_id or video_name is invalid.
        """

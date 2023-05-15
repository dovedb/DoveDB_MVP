

class OnlineInference(object):
    """Online inference module. Use spark streaming to calculate."""

    def __init__(self):
        pass

    def run_batch_inference(self, images):
        """Run inference on a batch of images using spark streaming.
        
        Args:
            images (list): A list of images.
        
        Returns:
            list: A list of inference result.
        """

    def load_event(self, event_id):
        """Load an event, compile the event model into a spark streaming 
            job.

        Args:
            event_id (int): Event ID.
        
        Returns:
            bool: True if success, otherwise False.
        
        Raises:
            ValueError: If event_id is invalid.
        """

    def remove_event(self, event_id):
        """Remove an event.

        Args:
            event_id (int): Event ID.
        
        Returns:
            bool: True if success, otherwise False.
        
        Raises:
            ValueError: If event_id is invalid.
        """


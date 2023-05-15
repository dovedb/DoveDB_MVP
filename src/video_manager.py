

class VideoManager(object):
    """Video manager."""

    def __init__(self):
        pass

    def add_frames(self, video_name, frames):
        """Add frames to a video.
        
        Args:
            video_name (str): Video name.
            frames (list): a list of opencv RGB image.
            
        Returns:
            bool: True if success, otherwise False.
        
        Raises:
            ValueError: If video_name is invalid.
        """
    
    def flush(self, video_name):
        """Flush a video from buffer to disk.
        
        Args:
            video_name (str): Video name.
        
        Returns:
            bool: True if success, otherwise False.
        
        Raises:
            ValueError: If video_name is invalid.
        """

    def read_frame(self, video_name, frame_id):
        """Read a frame from a video.
        
        Args:
            video_name (str): Video name.
            frame_id (int): Frame ID.
        
        Returns:
            opencv RGB image.
        
        Raises:
            ValueError: If video_name or frame_id is invalid.
            IndexError: If frame_id is out of range.
        """

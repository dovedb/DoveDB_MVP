import cv2
import numpy as np


class VideoManager(object):
    """Video manager."""

    def __init__(self):
        self.videos = {}
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') # XVID codec
        self.fps = 30.0 # 30 frames per second
        self.frame_size = (640, 480) # VGA resolution

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
        if not isinstance(video_name, str):
            raise ValueError('Invalid video name.')

        if video_name not in self.videos:
            self.videos[video_name] = []

        for frame in frames:
            if frame.shape != self.frame_size + (3,):
                return False
            self.videos[video_name].append(frame)
        
        return True
    
    def flush(self, video_name):
        """Flush a video from buffer to disk.
        
        Args:
            video_name (str): Video name.
        
        Returns:
            bool: True if success, otherwise False.
        
        Raises:
            ValueError: If video_name is invalid.
        """
        if not isinstance(video_name, str) or video_name not in self.videos:
            raise ValueError('Invalid video name.')

        out = cv2.VideoWriter(video_name+'.avi', self.fourcc, self.fps, self.frame_size)
        for frame in self.videos[video_name]:
            out.write(frame)
        out.release()

        # Delete the video data from memory.
        del self.videos[video_name]

        return True

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
        if not isinstance(video_name, str) or video_name not in self.videos:
            raise ValueError('Invalid video name.')
        if not isinstance(frame_id, int) or frame_id < 0 or frame_id >= len(self.videos[video_name]):
            raise ValueError('Invalid frame ID.')
        
        return self.videos[video_name][frame_id]

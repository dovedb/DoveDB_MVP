from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, TimestampType, LongType
import os
import torch

class OnlineInference(object):
    """Online inference module. Use spark streaming to calculate."""

    def __init__(self, frame_directory):
        
        self.spark = SparkSession.builder \
                .appName("VideoFrameProcessing") \
                .getOrCreate()
        self.image_schema = StructType([
            StructField("path", StringType(), False),
            StructField("modificationTime", TimestampType(), False),
            StructField("length", LongType(), False),
            StructField("content", BinaryType(), True)
            ])
        
        self.frame_stream = self.spark.readStream \
            .format("binaryFile") \
            .option("pathGlobFilter", "*.jpg") \
            .schema(self.image_schema) \
            .load(frame_directory)

    def run_batch_inference(self, images):
        """Run inference on a batch of images using spark streaming.
        
        Args:
            images (list): A list of images.
        
        Returns:
            list: A list of inference result.
        """
        self.frame_stream.writeStream \
            .foreachBatch(lambda frame_df, batch_id: self.process_frame(frame_df)) \
            .start() \
            .awaitTermination()

    def process_frame(self,frames):
        model = torch.hub("ultralytics/yolov5", "yolov5s", pretrained=True)
        detetions = model.inference(frames)
        
    
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


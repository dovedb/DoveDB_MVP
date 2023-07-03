from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from PIL import Image
import io
import os
import json

class OnlineInference(object):
    """Online inference module. Use spark streaming to calculate."""

    def __init__(self, frame_directory):
        self.frame_directory = frame_directory
        self.sc = SparkContext(appName="ImageProcessing")
        self.ssc = StreamingContext(self.sc, batchDuration=1)
        self.kafka_params = {
            "bootstrap.servers": "localhost:9092",
            "group.id": "image-processing-group"
        }
        self.events = {}

    def read_image_binary(self, binary_data):
        image = Image.open(io.BytesIO(binary_data))
        return image

    def resize_image(self, image):
        new_size = (int(image.size[0] * 0.5), int(image.size[1] * 0.5))
        resized_image = image.resize(new_size)
        return resized_image

    def save_image(self, rdd):
        frames = rdd.collect()
        for i, frame in enumerate(frames):
            file_path = os.path.join(self.frame_directory, f'frame_{i}.jpg')
            frame.save(file_path)
        
    def process_frame(self, frames):
        frames = frames.map(lambda x: self.read_image_binary(x[1])).map(self.resize_image)
        frames.foreachRDD(self.save_image)

    def run_batch_inference(self, images):
        """Run inference on a batch of images using spark streaming.
        
        Args:
            images (list): A list of images.
        
        Returns:
            list: A list of inference result.
        """
        # Convert images to RDD
        images_rdd = self.sc.parallelize(images)

        # Run inference on each image
        results = images_rdd.map(lambda img: self.model.predict(img))

        # Return results collected from RDD
        return results.collect()

    def process_frame(self,frames):
        frames = frames.map(lambda x: self.read_image_binary(x[1])).map(self.resize_image)
        frames.foreachRDD(self.save_image)
    
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
        if event_id not in self.events:
            raise ValueError(f"Invalid event_id: {event_id}")
        
        # Create a DStream that reads from Kafka
        kafka_stream = KafkaUtils.createDirectStream(self.ssc, [self.events[event_id]], self.kafka_params)

        # Process the frames from the Kafka stream
        self.process_frame(kafka_stream)

        self.ssc.start()
        return True

    def remove_event(self, event_id):
        """Remove an event.

        Args:
            event_id (int): Event ID.
        
        Returns:
            bool: True if success, otherwise False.
        
        Raises:
            ValueError: If event_id is invalid.
        """
        if event_id not in self.events:
            raise ValueError(f"Invalid event_id: {event_id}")
        del self.events[event_id]
        return True

    def add_event(self, event_id, topic):
        if event_id in self.events:
            raise ValueError(f"Event ID {event_id} already exists.")
        self.events[event_id] = topic


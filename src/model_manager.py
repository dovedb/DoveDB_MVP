<<<<<<< HEAD
import random
import string
from mm import *
import lightning.pytorch as pl
=======

>>>>>>> 390e109fe984b3f250f00952730c9995b3521dc9

class ModelManager(object):
    """Model manager."""

    def __init__(self):
<<<<<<< HEAD
        self.model_list = {}
        self.trainer_list = {}

    def create_model(self, model_cls, model_weight_path=None):
=======
        pass

    def create_model(self, model_cls, model_weight_path):
>>>>>>> 390e109fe984b3f250f00952730c9995b3521dc9
        """Create an new pretrained model.
        
        Args:
            model_cls (str): Model class name.
            model_weight_path (str): Model weight path.
        
        Returns:
            int: model ID.
        """
<<<<<<< HEAD
        model = build_network(model_cls, model_weight_path=model_weight_path)
        trainer = pl.Trainer()
        model_id = self._generate_model_id()
        self.model_list[model_id] = model
        self.trainer_list[model_id] = trainer
        return model_id

    def _generate_model_id(self):
        """Generate a random model ID.
        
        Returns:
            int: Model ID.
        """
        model_id = random.randint(0, 2**32)
        while model_id in self.model_list:
            model_id = random.randint(0, 2**32)
        return model_id

    def train_model(self, model_id):
        """Train a model.
        
        Args:
            model_id (int): Model ID.
        
        Returns:
            bool: True if success, otherwise False.

        Raises:
            ValueError: If model_id is invalid.
        """
=======
>>>>>>> 390e109fe984b3f250f00952730c9995b3521dc9

    def delete_model(self, model_id):
        """Delete a model.

        Args:
            model_id (int): Model ID.

        Returns:
            bool: True if success, otherwise False.

        Raises:
            ValueError: If model_id is invalid.
        """

    def deploy_model(self, model_id, device):
        """Deploy a model on a device.
        
        Args:
            model_id (int): Model ID.
            device (str): Device name.
        
        Returns:
            bool: True if success, otherwise False.

        Raises:
            ValueError: If model_id or device is invalid.
        """
<<<<<<< HEAD
        model = self.model_list[model_id]
        model.to(device)
        return True
=======
>>>>>>> 390e109fe984b3f250f00952730c9995b3521dc9

    def inference(self, model_id, *args, **kargs):
        """Model inference.
        
        Args:
            model_id (int): Model ID.
            *args: Model input.
            **kargs: Model input.
        
        Returns:
            Model output.
        
        Raises:
            ValueError: If model_id is invalid.
        """
<<<<<<< HEAD
        image = args[0]
        model = self.model_list[model_id]
        if not isinstance(image, list): image = [image]
        return model.predict_step(image)

if __name__ == "__main__":
    import cv2
    mmclass = ModelManager()
    model_id = mmclass.create_model("detr_m_detect")
    device = "cuda:0"
    flag = mmclass.deploy_model(model_id, device)
    if not flag:
        print("Deploy model failed.")
    
    image = cv2.imread("demo.jpg")
    results = mmclass.inference(model_id, image)
    detect_image = results[0].plot()
    cv2.imwrite("detected_demo.jpg", detect_image)
=======
>>>>>>> 390e109fe984b3f250f00952730c9995b3521dc9

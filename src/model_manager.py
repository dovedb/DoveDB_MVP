import random

from tqdm import tqdm
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from mm import *
from mm.mmdataset import build_yolo_dataset, IterableSimpleNamespace
from mm.mmdataloader import build_dataloader, build_mm_dataloader
from mm.mmchecks import check_det_dataset

class ModelManager(object):
    """Model manager."""

    def __init__(self):
        self.model_list = {}
        self.trainer_list = {}

    def create_model(self, model_cls, model_weight_path=None):
        """Create an new pretrained model.
        
        Args:
            model_cls (str): Model class name.
            model_weight_path (str): Model weight path.
        
        Returns:
            int: model ID.
        """
        model = build_network(model_cls, model_weight_path=model_weight_path)
        trainer = pl.Trainer(log_every_n_steps=5, max_epochs=model.args['epochs'])
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
        if "yolo" in self.model_list[model_id].model_name:
            print("Start training model {}.".format(model_id))
            current_file = os.path.realpath(__file__)
            current_directory = os.path.dirname(current_file)
            config_file = os.path.join(current_directory, "mm/configs", "coco.yaml")
            gs = max(int(self.model_list[model_id].model.stride.max() if self.model_list[model_id] else 0), 32)
            model_cfg = self.model_list[model_id].model.args.copy()
            for k, v in model_cfg.items():
                if isinstance(v, str) and v.lower() == 'none':
                    model_cfg[k] = None
            model_cfg = IterableSimpleNamespace(**model_cfg)
            data_cfg = check_det_dataset(config_file)
            dataset = build_yolo_dataset(model_cfg, f"./cache/{self.model_list[model_id].model_type}/images",
                                        self.model_list[model_id].args['batch'], data_cfg, 
                                        mode='train', rect=False, stride=gs)
            dataloader = build_dataloader(dataset, self.model_list[model_id].args['batch'],
                                        self.model_list[model_id].args['workers'],
                                        self.model_list[model_id].args['shuffle'])
            self.trainer_list[model_id].fit(self.model_list[model_id], dataloader)
        elif "fast" or "sdd" or "detr" in self.model_list[model_id].model_name:
            print("Start training model {}.".format(model_id))
            dataloader = build_mm_dataloader(self.model_list[model_id].model_size)
            self.model_list[model_id].train()
            self.trainer_list[model_id].fit(self.model_list[model_id], dataloader)
        
        return True    
        
    def delete_model(self, model_id):
        """Delete a model.

        Args:
            model_id (int): Model ID.

        Returns:
            bool: True if success, otherwise False.

        Raises:
            ValueError: If model_id is invalid.
        """
        del self.model_list[model_id]
        del self.trainer_list[model_id]
        return True

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
        self.model_list[model_id].model.to(device)
        return True

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
        image = args[0]
        model = self.model_list[model_id]
        if not isinstance(image, list): image = [image]
        return model.predict_step(image)

if __name__ == "__main__":
    import cv2
    mmclass = ModelManager()
    model_id = mmclass.create_model("ssd_s_detect")
    device = "cuda:0"
    flag = mmclass.deploy_model(model_id, device)
    if not flag:
        print("Deploy model failed.")
    
    image = cv2.imread("demo.jpg")
    results = mmclass.inference(model_id, image)
    
    flag = mmclass.train_model(model_id)
    assert flag, "Train model failed."

    device = "cuda:1"
    mmclass.deploy_model(model_id, device)    
    image = cv2.imread("demo.jpg")
    results = mmclass.inference(model_id, image)
    
    # delete model and trainer
    flag = mmclass.delete_model(model_id)
    assert flag, "Delete model failed."
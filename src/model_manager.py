

class ModelManager(object):
    """Model manager."""

    def __init__(self):
        pass

    def create_model(self, model_cls, model_weight_path):
        """Create an new pretrained model.
        
        Args:
            model_cls (str): Model class name.
            model_weight_path (str): Model weight path.
        
        Returns:
            int: model ID.
        """

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

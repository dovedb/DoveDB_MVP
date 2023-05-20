import os
import torch
import numpy as np

from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
from mmdet.evaluation.functional import coco_classes
from mmdet.models.data_preprocessors import DetDataPreprocessor

from ..mmresult import Results
from ..mmutils import build_annotations, is_list_of

CFG_DICT = {
    "s": "ssd_small.py",
    "m": "ssd_medium.py",
    "l": "ssd_large.py"
}

CFG_B_DICT = {
    "ssd_small.py": "s",
    "ssd_medium.py": "m",
    "ssd_large.py": "l"
}


def build_ssd(current_directory,
              model_name, model_size,
              model_weight_path, input_array=True):
    assert model_size in CFG_DICT.keys(), f'model_size must be {CFG_DICT.keys()}'
    cfg_path = os.path.join(current_directory, "configs", "ssd", CFG_DICT[model_size])
    assert model_size in model_weight_path, \
    f"{model_weight_path.replace('-', '_')} must contain {CFG_DICT[model_size].rstrip('.py').replace('-', '_')}"
    model = init_detector(cfg_path, model_weight_path)
    cfg = model.cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    if input_array:
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline)
    class_names = coco_classes()
    
    def preprocess(cls, **kwargs):
        img = kwargs["orig_imgs"]
        data_ = dict(img=img, img_id=0)
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        return data_

    def postprocess(cls, **kwargs):
        preds = kwargs["preds"]
        orig_imgs = kwargs["orig_imgs"]
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            new_pred = []
            for bbox, label, score in zip(pred.pred_instances.bboxes,
                                          pred.pred_instances.labels,
                                          pred.pred_instances.scores):
                bbox, label, score = bbox.cpu().numpy(), label.cpu().numpy(), score.cpu().numpy()
                if score < cls.args['conf']: continue
                new_pred.append([bbox[0], bbox[1], bbox[2], bbox[3], score, label])
            new_pred = np.array(new_pred)
            results.append(Results(orig_img=orig_img, path="./", names=class_names, boxes=new_pred))
        return results
    
    annotation_path = os.path.join(cfg.train_dataloader.dataset.dataset.data_root, "annotations/train.json")
    if not os.path.exists(annotation_path):
        build_annotations(cfg, annotation_path, class_names)
    
    def preprocess_batch(cls, **kwargs):
        batch = kwargs["batch"]
        if not hasattr(cls, "data_processor"):
            cls.data_processor = DetDataPreprocessor(cfg.model.data_preprocessor.mean,
                                                     cfg.model.data_preprocessor.std).to(cls.device)
            processed_data = cls.data_processor(batch, training=True)
            processed_data["inputs"] = processed_data["inputs"].to(cls.device)
            processed_data["data_samples"] = [data_sample.to(cls.device) for data_sample in processed_data["data_samples"]]
            return processed_data
        processed_data = cls.data_processor(batch, training=True)
        processed_data["inputs"] = processed_data["inputs"].to(cls.device)
        processed_data["data_samples"] = [data_sample.to(cls.device) for data_sample in processed_data["data_samples"]]
        return processed_data
    
    def loss_function(cls, **kwargs):
        batch = kwargs["batch"]
        losses = cls.model.loss(batch["inputs"], batch["data_samples"])
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
        loss = sum(value for key, value in log_vars if 'loss' in key)
        return loss
        
    return model, \
           preprocess, postprocess, \
           preprocess_batch, loss_function
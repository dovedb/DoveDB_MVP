import os
import numpy as np

from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
from mmdet.evaluation.functional import coco_classes

from ..mmresult import Results

CFG_DICT = {
    "s": "ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py",
    "m": "ssd300_coco.py",
    "l": "ssd512_coco.py"
}

CFG_B_DICT = {
    "ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py": "s",
    "ssd300_coco.py": "m",
    "ssd512_coco.py": "l"
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
        
    return model, preprocess, postprocess
import os
import torch
import numpy as np
import lightning.pytorch as pl

from .modules.yolo import build_yolov
from .modules.fasterrcnn import build_faster_rcnn
from .modules.ssd import build_ssd
from .modules.detr import build_detr

SUPPORTED_DETECTORS = ["yolov8", "yolov5", "yolov3", "fasterrcnn", "ssd", "detr"]
SUPPORTED_TRACKERS = ["sort", "deepsort", "bytetrack"]

PRETRAIN_WEIGHT_DICT = {"yolov8": {"n": "./mm/weights/yolov8n.pt",
                                   "s": "./mm/weights/yolov8s.pt",
                                   "m": "./mm/weights/yolov8m.pt",
                                   "l": "./mm/weights/yolov8l.pt",
                                   "x": "./mm/weights/yolov8x.pt"},
                        "yolov5": {"n": "./mm/weights/yolov5n.pt",
                                   "s": "./mm/weights/yolov5s.pt",
                                    "m": "./mm/weights/yolov5m.pt",
                                    "l": "./mm/weights/yolov5l.pt",
                                    "x": "./mm/weights/yolov5xu.pt"},
                        "yolov3": {"x": "./mm/weights/yolov3xu.pt"},
                        "fasterrcnn": {"s": "./mm/weights/fasterrcnn_s.pth",
                                       "m": "./mm/weights/fasterrcnn_m.pth",
                                       "l": "./mm/weights/fasterrcnn_l.pth"},
                        "ssd": {"s": "./mm/weights/ssd_s.pth",
                                "m": "./mm/weights/ssd_m.pth",
                                "l": "./mm/weights/ssd_l.pth"},
                        "detr": {"m": "./mm/weights/detr_m.pth"}}

class MMNetwork(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = kwargs["model"]
        self.model_type = kwargs["model_type"]
        self.model_name = kwargs["model_name"]
        self.model_size = kwargs["model_size"]
        self.model_weight_path = kwargs["model_weight_path"]
        
        self.preprocess = kwargs.get("preprocess", lambda x: x)
        self.postprocess = kwargs.get("postprocess", lambda x: x)
        
        self.args = kwargs["args"]
    
    @torch.no_grad()
    def predict_step(self, im0s):
        if "yolo" in self.model_name:
            if not isinstance(im0s, torch.Tensor): ims = self.preprocess(self, 
                                                                       orig_imgs=im0s)
            else: ims = im0s
            preds = self.model(ims)[0]
            results = self.postprocess(self, preds=preds,
                                       img=ims, orig_imgs=im0s)
        elif "fasterrcnn" or "ssd" or "detr" in self.model_name:
            preds = []
            for im0 in im0s:
                im = self.preprocess(self, orig_imgs=im0)
                pred = self.model.test_step(im)[0]
                preds.append(pred)
            results = self.postprocess(self, preds=preds, 
                                       orig_imgs=im0s)
        return results
    
    @property
    def device(self):
        device = next(self.parameters()).device
        return device

def build_detector(model_name, model_size, model_weight_path):
    current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file)
    
    if model_weight_path is None and model_name in PRETRAIN_WEIGHT_DICT and model_size in PRETRAIN_WEIGHT_DICT[model_name]:
        model_weight_path = PRETRAIN_WEIGHT_DICT[model_name][model_size]
    
    print(f"Building {model_weight_path} model...")
    if "yolo" in model_name: detector, preprocess, postprocess = build_yolov(current_directory,
                                                                             model_name, model_size,
                                                                             model_weight_path)
    elif "fast" in model_name: detector, preprocess, postprocess = build_faster_rcnn(current_directory,
                                                                                     model_name, model_size,
                                                                                     model_weight_path)
    elif "ssd" in model_name: detector, preprocess, postprocess = build_ssd(current_directory,
                                                                            model_name, model_size,
                                                                            model_weight_path)
    elif "detr" in model_name: detector, preprocess, postprocess = build_detr(current_directory,
                                                                              model_name, model_size,
                                                                              model_weight_path)
    
    model_info = {
        "model": detector,
        "model_type": "detect",
        "model_name": model_name,
        "model_size": model_size,
        "model_weight_path": model_weight_path,
        "args" : {"half": False,
                  "conf": 0.75,
                  "iou": 0.45,
                  "imgsz": 640},
        "preprocess": preprocess,
        "postprocess": postprocess
    }
    
    network = MMNetwork(**model_info)
    
    return network

def build_detector_trainer():
    pass

def build_tracker():
    pass

def build_network(model_cls,
                  model_weight_path=None):
    model_cls = model_cls.lower().split("_")
    model_type = None
    if len(model_cls) == 3: model_name, model_size, model_type = model_cls
    else: (model_name, model_size), model_type = model_cls, "detect" if model_cls[0] in SUPPORTED_DETECTORS else "track"
    print(model_name, SUPPORTED_DETECTORS + SUPPORTED_TRACKERS)
    assert model_name in SUPPORTED_DETECTORS + SUPPORTED_TRACKERS, f"Unsupported model name: {model_name}"
    assert model_type in ["detect", "track"], f"Unsupported model type: {model_type}"
    
    if model_type == "detect":
        model = build_detector(model_name, model_size, model_weight_path)
    elif model_type == "track":
        model = build_tracker(model_name, model_size, model_weight_path)
        
    return model
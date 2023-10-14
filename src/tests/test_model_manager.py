import cv2
import pytest
import random
import model_manager as mm
import pycuda.driver as cuda

PRETRAIN_WEIGHT_DICT = {
                        "yolov8": {
                                   "n": "./mm/weights/yolov8n.pt",
                                   "s": "./mm/weights/yolov8s.pt",
                                   "m": "./mm/weights/yolov8m.pt",
                                   "l": "./mm/weights/yolov8l.pt",
                                   "x": "./mm/weights/yolov8x.pt"
                                   },
                        "yolov5": {
                                   "n": "./mm/weights/yolov5n.pt",
                                   "s": "./mm/weights/yolov5s.pt",
                                   "m": "./mm/weights/yolov5m.pt",
                                   "l": "./mm/weights/yolov5l.pt",
                                   "x": "./mm/weights/yolov5xu.pt"
                                   },
                        "yolov3": {
                                   "x": "./mm/weights/yolov3u.pt"
                                   },
                        "fasterrcnn": {
                                    "s": "./mm/weights/fasterrcnn_s.pth",
                                    "m": "./mm/weights/fasterrcnn_m.pth",
                                    "l": "./mm/weights/fasterrcnn_l.pth"
                                    },
                        "ssd": {
                                "s": "./mm/weights/ssd_s.pth",
                                "m": "./mm/weights/ssd_m.pth",
                                "l": "./mm/weights/ssd_l.pth"
                                },
                        "detr": {
                                "m": "./mm/weights/detr_m.pth"
                            }}

pytest_model_info_list = []
for device_id in range(2):
    for model_name in PRETRAIN_WEIGHT_DICT.keys():
        for model_size in PRETRAIN_WEIGHT_DICT[model_name].keys():
            pytest_model_info_list.append([f"{model_name}_{model_size}", device_id])
for device_id in range(2):
    for model_name in PRETRAIN_WEIGHT_DICT.keys():
        for model_size in PRETRAIN_WEIGHT_DICT[model_name].keys():
            pytest_model_info_list.append([f"{model_name}_{model_size}_detect", device_id])


@pytest.mark.parametrize(
    "model_info, device_id", 
    pytest_model_info_list,
)
def test_add(model_info, device_id):
    mmclass = mm.ModelManager()
    model_id = mmclass.create_model(model_info)
    device = f"cuda:{device_id}"
    flag = mmclass.deploy_model(model_id, device)
    assert flag == True, "Deploy model failed."
    
    image = cv2.imread("demo.jpg")
    results = mmclass.inference(model_id, image)
    detected_image = results[0].plot()
    cv2.imwrite(f"{model_info}.jpg", detected_image)
    
    flag = mmclass.train_model(model_id)
    assert flag == True, "Train model failed."

    device_id = abs(device_id - 1)
    device = f"cuda:{device_id}"
    mmclass.deploy_model(model_id, device)    
    image = cv2.imread("demo.jpg")
    results = mmclass.inference(model_id, image)
    detected_image = results[0].plot()
    cv2.imwrite(f"{model_info}.jpg", detected_image)
    
    flag = mmclass.train_model(model_id)
    assert flag == True, "Train model failed."
    
    # delete model and trainer
    flag = mmclass.delete_model(model_id)
    assert flag == True, "Delete model failed."

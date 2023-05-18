import cv2
import pytest
import model_manager as mm

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
                        "yolov3": {"x": "./mm/weights/yolov3u.pt"},
                        "fasterrcnn": {"s": "./mm/weights/fasterrcnn_s.pth",
                                       "m": "./mm/weights/fasterrcnn_m.pth",
                                       "l": "./mm/weights/fasterrcnn_l.pth"},
                        "ssd": {"s": "./mm/weights/ssd_s.pth",
                                "m": "./mm/weights/ssd_m.pth",
                                "l": "./mm/weights/ssd_l.pth"},
                        "detr": {"m": "./mm/weights/detr_m.pth"}}

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
    if not flag:
        print("Deploy model failed.")
    
    image = cv2.imread("demo.jpg")
    results = mmclass.inference(model_id, image)
    detect_image = results[0].plot()
    flag = cv2.imwrite(f"detected_demo_{model_info}.jpg", detect_image)
    assert flag == True
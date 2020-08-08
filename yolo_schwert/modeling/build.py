from yolo_schwert.modeling.detectors import YOLOv3


def build_model(cfg, cuda=True):
    if cfg.MODEL.TYPE == "YOLOv3":
        model = YOLOv3(cfg.MODEL, cuda)
    return model
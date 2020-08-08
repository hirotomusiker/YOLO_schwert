from yolo_schwert.data.cocodataset import COCODataset
from .coco_evaluator import COCOEvaluator


def build_evaluator(dataset_name, cfg):
    if dataset_name == 'COCO':
        dataset = COCODataset(cfg, train=False)
        return COCOEvaluator(
            dataset,
            img_size=cfg.TEST.IMGSIZE,
            confthre=cfg.TEST.CONFTHRE_AP,
            nmsthre=cfg.TEST.NMSTHRE,
            num_classes = cfg.MODEL.N_CLASSES,
        )
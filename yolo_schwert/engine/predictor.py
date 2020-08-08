import torch
from yolo_schwert.modeling.build import build_model
from yolo_schwert.checkpoint import YOLOCheckpointer
from yolo_schwert.utils.yolo_process import preprocess, postprocess, yolobox2label
from yolo_schwert.utils.coco_utils import get_coco_label_names


class Predictor:
    """ Predictor class that holds a model and perform inference on an image.
    """
    def __init__(self, cfg, cuda=False):
        self.device = 'cuda' if cuda else 'cpu'
        self.model = build_model(cfg, cuda).to(self.device)
        self.num_classes = cfg.MODEL.N_CLASSES
        self.confthre = cfg.TEST.CONFTHRE
        self.nmsthre = cfg.TEST.NMSTHRE
        self.checkpointer = YOLOCheckpointer(
            self.model,
        )
        self.img_size = cfg.TEST.IMGSIZE
        if cfg.MODEL.WEIGHTS:
            _ = self.checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()
        self.coco_class_names = coco_class_names
        self.coco_class_ids = coco_class_ids
        self.coco_class_colors = coco_class_colors

    def predict_one_image(self, img):
        """
        perform inference on one image.
        Args:
            img (np.array): image data (BGR) read by cv2.imread
        Returns:
            boxes (list): bounding box data in [y1, x1, y2, x2] order
            classes (list): coco class ids
            colors (list): class-id specific color for box visualization.
        """
        img, info_img = preprocess(img, self.img_size, jitter=0,
                                   random_placing=False,
                                   random_crop=False)
        img = torch.tensor(img / 255.).permute((2, 0, 1)).unsqueeze(0)
        img = img.to(self.device).float()
        with torch.no_grad():
            outputs = self.model(img, None)
        outputs = postprocess(
            outputs, self.num_classes, self.confthre, self.nmsthre)

        if outputs[0] is None:
            print("No Objects Deteted!!")
            return

        bboxes = list()
        classes = list()
        colors = list()

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
            cls_id = self.coco_class_ids[int(cls_pred)]
            print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
            print('\t+ Label: %s, Conf: %.5f' %
                  (self.coco_class_names[cls_id], cls_conf.item()))
            box = yolobox2label([y1, x1, y2, x2], info_img)
            bboxes.append(box)
            classes.append(cls_id)
            colors.append(self.coco_class_colors[int(cls_pred)])
        return bboxes, classes, colors
from collections import defaultdict
from torch import nn
from yolo_schwert.modeling.backbones import build_darknet
from yolo_schwert.modeling.necks import build_yolov3_neck
from yolo_schwert.modeling.dense_heads import build_yolo_layer

class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self, config_model, cuda=True):
        """
        Initialization of YOLOv3 class.
        Args:
            config_model : config.MODEL
            cuda (bool) : use GPU or not
        """
        super(YOLOv3, self).__init__()
        self.backbone = build_darknet(config_model)
        self.neck = build_yolov3_neck(config_model,
                                      in_channels=[1024, 512, 256])
        self.dense_head = build_yolo_layer(config_model,
                                           in_channels=[1024, 512, 256],
                                           cuda=cuda)


    def forward(self, x, targets=None):
        """
        Forward path of YOLOv3.
        """
        x = self.backbone(x)
        x_layers = self.neck(x)  # 3 layers -> 3 layers
        output = self.dense_head(x_layers, targets) # 3 layers -> 1 output loss dict or tensor
        return output
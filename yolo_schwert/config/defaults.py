from fvcore.common.config import CfgNode as CN

_C = CN()

#######################
# Model Configuration #
#######################

_C.MODEL = CN()
_C.MODEL.TYPE = "YOLOv3"
_C.MODEL.BACKBONE = "darknet53"
_C.MODEL.ANCHORS = [[10, 13], [16, 30], [33, 23],
          [30, 61], [62, 45], [59, 119],
          [116, 90], [156, 198], [373, 326]]
_C.MODEL.ANCH_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
_C.MODEL.ACT = "lrelu"
_C.MODEL.N_CLASSES = 80
_C.MODEL.XYWH_MODE = "YOLOv3"
_C.MODEL.LOSS_REDUCTION = 'sum'
_C.MODEL.WEIGHTS = ""
_C.MODEL.IGNORETHRE = 0.7

# New Features
_C.MODEL.SPP = False
_C.MODEL.EFFNET_NAME = ""
_C.MODEL.EFFNET_WEIGHTS = ""
_C.MODEL.GIOU_LOSS = False
_C.MODEL.GIOU_WEIGHT = 0.05
_C.MODEL.GIOU_CONF = False
_C.MODEL.IOU_THRE = 0.2
_C.MODEL.TARGET_MODE = "YOLOv3"
_C.MODEL.CROSS_TARGET = False
_C.MODEL.BOX_TARGET = False
_C.MODEL.NEIGHBOUR_DIST = 0.5

########################
# Solver Configuration #
########################

_C.SOLVER = CN()
_C.SOLVER.SCHEDULER = 'WarmUpMultiStepLR'
_C.SOLVER.LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.DECAY = 0.0005
_C.SOLVER.BURN_IN = 1000
_C.SOLVER.MAX_ITER = 500000
_C.SOLVER.STEPS = (400000, 450000)
_C.SOLVER.BATCHSIZE = 4
_C.SOLVER.SUBDIVISION = 16

#############################
# Input Image Configuration #
#############################

_C.INPUT = CN()
_C.INPUT.IMG_SIZES = (416,)
_C.INPUT.IMGSIZE = 608

#######################
# Dataloader Settings #
#######################
_C.DATA = CN()
_C.DATA.ROOT_DIR = "COCO"
_C.DATA.TRAIN_DIR = "train2017"
_C.DATA.VAL_DIR = "val2017"
_C.DATA.TRAIN_JSON = 'instances_train2017.json'
_C.DATA.VAL_JSON = 'instances_val2017.json'

#########################
# Augmentation Settings #
#########################
_C.AUG = CN()
_C.AUG.JITTER = 0.3
_C.AUG.BOX_JITTER = 0.0
_C.AUG.RANDOM_PLACING = True
_C.AUG.RANDOM_RESIZE = True
_C.AUG.HUE = 0.1
_C.AUG.SATURATION = 1.5
_C.AUG.EXPOSURE = 1.5
_C.AUG.LRFLIP = True
_C.AUG.RANDOM_DISTORT = False

# New Augmentations
_C.AUG.MOSAIC = False
_C.AUG.TRANS = False
_C.AUG.RANDOM_CROP = False
_C.AUG.RANDRESIZE = True
_C.AUG.VFLIP = False

######################
# Test Configuration #
######################
_C.TEST = CN()
_C.TEST.CONFTHRE = 0.6
_C.TEST.CONFTHRE_AP = 0.005
_C.TEST.NMSTHRE = 0.45
_C.TEST.IMGSIZE = 416
_C.TEST.INTERVAL = 2000

##########
# Others #
##########
_C.OUTPUT_DIR = "results"
_C.NUM_WORKERS = 0
_C.LOG_INTERVAL = 10
_C.CHECKPOINT_INTERVAL = 100

import argparse
import torch
import cv2

import _init_paths
from yolo_schwert.engine.predictor import Predictor
from yolo_schwert.config import get_cfg
from yolo_schwert.utils.coco_utils import get_coco_label_names

def parse_args():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--config", default="", metavar="FILE", type=str, help="path to config")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,)
    parser.add_argument('--image', type=str)
    parser.add_argument('--background', action='store_true',
                        default=False, help='background(no-display mode. save "./output.png")')
    parser.add_argument('--detect_thresh', type=float,
                        default=None, help='confidence threshold')
    return parser.parse_args()

def main():

    args = parse_args()
    print("Setting Arguments.. : ", args)
    cuda = torch.cuda.is_available()
    #os.makedirs(args.checkpoint_dir, exist_ok=True)

    cfg = get_cfg()
    if args.config:
        cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    print("successfully loaded config file: ", cfg)

    predictor = Predictor(cfg, cuda=cuda)

    img = cv2.imread(args.image)
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))

    bboxes, classes, colors = predictor.predict_one_image(img)

    if args.background:
        import matplotlib
        matplotlib.use('Agg')

    from yolo_schwert.utils.vis_bbox import vis_bbox
    import matplotlib.pyplot as plt

    vis_bbox(
        img_raw, bboxes, label=classes, label_names=predictor.coco_class_names,
        instance_colors=colors, linewidth=2)
    plt.show()

    if args.background:
        plt.savefig('output.png')


if __name__ == '__main__':
    main()

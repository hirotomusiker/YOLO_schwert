import _init_paths
import argparse
from yolo_schwert.engine.trainer import Trainer
from yolo_schwert.data.cocodataset import *


from yolo_schwert.config import get_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", metavar="FILE", type=str, help="path to config")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    return parser.parse_args()


def main():
    """
    YOLOv3 trainer. See README for details.
    """
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

    trainer = Trainer(cfg, cuda=cuda)
    trainer.train()


if __name__ == '__main__':
    main()

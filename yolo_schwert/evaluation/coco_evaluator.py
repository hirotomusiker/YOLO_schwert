import json
import tempfile
from tqdm import tqdm

from pycocotools.cocoeval import COCOeval
import torch
from yolo_schwert.utils.yolo_process import postprocess, yolobox2label


class COCOEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, dataset, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0)
        self.img_size = img_size
        self.confthre = confthre # 0.005 (darknet)
        self.nmsthre = nmsthre # 0.45 (darknet)
        self.num_classes = num_classes
        cuda = torch.cuda.is_available()
        self.device = 'cuda' if cuda else 'cpu'

    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()

        ids = []
        data_dict = []
        dataiterator = iter(self.dataloader)
        for (img, _, info_img, id_) in tqdm(dataiterator):
            info_img = [float(info) for info in info_img]
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                img = img.float().to(self.device)
                outputs = model(img)
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre)

                if outputs[0] is None:
                    continue
                outputs = outputs[0].cpu().data

            for output in outputs:
                x1 = float(output[0])
                y1 = float(output[1])
                x2 = float(output[2])
                y2 = float(output[3])
                label = self.dataset.class_ids[int(output[6])]
                box = yolobox2label((y1, x1, y2, x2), info_img)
                bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                score = float(output[4].data.item() * output[5].data.item()) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score, "segmentation": []} # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            return self.parse_stats_to_dict(cocoEval.stats)
        else:
            return {}

    def parse_stats_to_dict(self, stats):
        keys = {'AP50:95', 'AP50', 'AP75'
                , 'AP50:95_small', 'AP50:95_medium', 'AP50:95_large'
        } # drop ARs
        results = {'AP50:95': stats[0],
                   'AP50': stats[1],
                   'AP75': stats[2],
                   'AP50:95_small': stats[3],
                   'AP50:95_medium': stats[4],
                   'AP50:95_large': stats[5]}
        return results



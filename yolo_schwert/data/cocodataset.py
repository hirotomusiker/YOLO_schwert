import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from pycocotools.coco import COCO

from yolo_schwert.utils.yolo_process import preprocess, label2yolobox
from yolo_schwert.data.transforms import random_distort


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, cfg, train=False, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        """
        # TODO: add transform modules
        img_dir = cfg.DATA.TRAIN_DIR if train else cfg.DATA.VAL_DIR
        annot_file = cfg.DATA.TRAIN_JSON if train else cfg.DATA.VAL_JSON
        self.data_dir = Path(cfg.DATA.ROOT_DIR).joinpath(img_dir)
        self.data_dir_2 = Path(cfg.DATA.ROOT_DIR).joinpath("train2017")
        annot_path = str(Path(cfg.DATA.ROOT_DIR).joinpath('annotations').joinpath(annot_file))

        self.coco = COCO(annot_path)
        self.train = train
        self.ids = self.coco.getImgIds()
        print(len(self.ids), 'images are registered')
        if debug:
            self.ids = self.ids[:32]
            print("debug mode...", self.ids)
        self.imgs = self.coco.loadImgs(self.ids)
        self.min_size = 1.
        self.img_size = cfg.INPUT.IMGSIZE
        self.max_labels = 500
        self.class_ids = sorted(self.coco.getCatIds())
        self.lrflip = cfg.AUG.LRFLIP if train else False
        self.vflip = cfg.AUG.VFLIP if train else False
        self.trans = cfg.AUG.TRANS if train else False
        self.jitter = cfg.AUG.JITTER if train else 0
        self.random_placing = cfg.AUG.RANDOM_PLACING if train else False
        self.random_crop = cfg.AUG.RANDOM_CROP if train else False
        self.hue = cfg.AUG.HUE
        self.saturation = cfg.AUG.SATURATION
        self.exposure = cfg.AUG.EXPOSURE
        self.random_distort = cfg.AUG.RANDOM_DISTORT if train else False
        self.box_jitter = cfg.AUG.BOX_JITTER if train else False
        self.mosaic = cfg.AUG.MOSAIC if train else False

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        lrflip = False
        trans = False
        vflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:
            lrflip = True
        if np.random.rand() > 0.5 and self.trans == True:
            trans = True
        if np.random.rand() > 0.5 and self.vflip == True:
            vflip = True

        # load image and preprocess
        img_file = self.data_dir.joinpath(self.imgs[index]['file_name'])
        img = cv2.imread(str(img_file))
        if img is None:
            img_file = self.data_dir_2.joinpath(self.imgs[index]['file_name'])
            img = cv2.imread(str(img_file))

        assert img is not None, 'cannot load {}'.format(img_file)

        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing,
                                   random_crop=self.random_crop)

        if self.random_distort:
            img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        if lrflip:
            img = np.flip(img, axis=2).copy()
        if vflip:
            img = np.flip(img, axis=1).copy()
        if trans:
            img = np.transpose(img, (0, 2, 1))

        # load labels
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(anno['bbox'])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            labels = label2yolobox(labels, info_img, self.img_size, lrflip)#, vflip, trans, self.box_jitter)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, id_


    # TODO: New Features. These will be implemented as modules later :)

    def flip(self, img, labels, img_size=1024):
        flag = 0
        if np.random.rand() > 0.5 and self.lrflip == True:
            img = np.flip(img, axis=1).copy()
            labels = [[l[0], img_size-l[1]-l[3], l[2], l[3], l[4]] for l in labels]
            flag += 1
        if np.random.rand() > 0.5 and self.vflip == True:
            img = np.flip(img, axis=0).copy()
            labels = [[l[0], l[1], img_size-l[2]-l[4], l[3], l[4]] for l in labels]
            flag += 10
        if np.random.rand() > 0.5 and self.trans == True:
            img = np.transpose(img, (1, 0, 2))
            labels = [[l[0], l[2], l[1], l[4], l[3]] for l in labels]
            flag += 100
        return img, labels, flag

    def filter_labels(self, labels, img_size=1024):
        # xywh-in, xyxy-out
        labels[:, 3] += labels[:, 1]
        labels[:, 4] += labels[:, 2]
        labels = np.clip(labels, 0, img_size)
        labels = [l for l in labels if (l[3] - l[1]) > self.min_size\
                and (l[4] - l[2]) > self.min_size\
                and (l[3] - l[1]) / (l[4] - l[2]) < self.max_aspect_ratio\
                and (l[4] - l[2]) / (l[3] - l[1]) < self.max_aspect_ratio]
        return labels

    def load_labels(self, id_):
        labels = []
        img_id = self.ids[id_]
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        for anno in annotations:
            labels.append([])
            labels[-1].append(anno['category_id'] - 1)
            labels[-1].extend([float(b) for b in anno['bbox']])
        return labels

    def load_multiple_images(self):
        # functionality for Mosaic Augmentation.
        # This will be moved to a mosaic dataset module
        size = 1024
        img_out = np.ones((int(size*3), int(size*3), 3), dtype=np.uint8) * 128
        all_labels = []
        ids = np.random.randint(len(self.ids), size=4)
        dxs = [size//2, size//2+size, size//2, size//2+size]
        dys = [size//2, size//2, size//2+size, size//2+size]
        for (id_, dx, dy)  in zip(ids, dxs, dys):
            path = os.path.join(self.data_dir, self.name,
                                self.imgs[id_]['file_name'] + '.jpg')
            img_out[dy:dy+size, dx:dx+size] = cv2.imread(path)
            labels = self.load_labels(id_)  # xywh
            labels = [[l[0], l[1]+dx, l[2]+dy, l[3], l[4]] for l in labels]
            all_labels.extend(labels)
        return img_out, all_labels

    def random_crop_from_multiple_images(self,
                                         img,
                                         labels,
                                         scale_min=0.5,
                                         scale_max=1.5,
                                         disp_max=0.2,
                                         dist_size=1024):
        # functionality for Mosaic Augmentation.
        # This will be moved to a mosaic dataset module
        scale = np.random.random() * (scale_max - scale_min) + scale_min
        crop_size = int(dist_size / scale)  # 2048 to 680
        # scale = 0,5: 0 to
        # scale = 1.5: 512 to 1280
        d_low = (2048 - crop_size) // 2
        d_hi =  3072 - crop_size - d_low
        dx = np.random.randint(d_low, d_hi)
        dy = np.random.randint(d_low, d_hi)
        img = img[dy:dy+crop_size, dx:dx+crop_size]
        labels = [[l[0], l[1]-dx, l[2]-dy, l[3], l[4]] for l in labels]
        mag = dist_size / crop_size
        img = cv2.resize(img, (dist_size, dist_size), interpolation=cv2.INTER_LINEAR)
        labels = [[l[0], l[1]*mag, l[2]*mag, l[3]*mag, l[4]*mag] for l in labels]

        return img, labels


    def mosaic(self):
        # Mosaic Augmentation.
        # This will be moved to a mosaic dataset module
        if self.mosaic and self.train:
            img, labels = self.load_multiple_images()
            img, labels = self.random_crop_from_multiple_images(img, labels)
            img, labels, flag = self.flip(img, labels)
            padded_labels = np.zeros((self.max_labels, 5))
            if len(labels) > 0:
                labels = np.stack(labels)
                labels = self.filter_labels(labels)
            if len(labels) > 0:
                labels = np.stack(labels)
                labels /= 1024.
                labels_ = labels.copy()
                labels_[:, 0] = 0
                labels_[:, 1] = (labels[:, 1] + labels[:, 3]) / 2
                labels_[:, 2] = (labels[:, 2] + labels[:, 4]) / 2
                labels_[:, 3] = (labels[:, 3] - labels[:, 1])
                labels_[:, 4] = (labels[:, 4] - labels[:, 2])
                padded_labels[range(len(labels_))[:self.max_labels]
                          ] = labels_[:self.max_labels]
            padded_labels = torch.from_numpy(padded_labels)

            img = np.asarray(img[:, :, ::-1], dtype=np.uint8)
            if self.random_distort:
                img = random_distort(img, self.hue, self.saturation, self.exposure)
            img = img.transpose(2, 0, 1) / 255.  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            return img, padded_labels, (0,), id_
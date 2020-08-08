import torch
import torch.nn as nn
import numpy as np
from fvcore.nn import giou_loss
from yolo_schwert.utils.boxes import xywh_to_xyxy, bboxes_iou, giou
from yolo_schwert.utils.target import add_neighbours


class YOLOLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(self, config_model, layer_no, in_ch, ignore_thre=0.7, giou_conf=0., cuda=True):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8]  # fixed
        self.anchors = config_model.ANCHORS
        self.anch_mask = config_model.ANCH_MASK[layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model.N_CLASSES
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(reduction=config_model.LOSS_REDUCTION)
        self.bce_loss = nn.BCELoss(reduction=config_model.LOSS_REDUCTION)
        self.stride = strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.cuda.FloatTensor(self.ref_anchors)
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)
        self.giou_loss = config_model.GIOU_LOSS
        self.giou_weight = config_model.GIOU_WEIGHT
        self.iou_thre = config_model.IOU_THRE
        self.target_assignment_mode = config_model.TARGET_MODE
        self.cross_target = config_model.CROSS_TARGET
        self.box_target = config_model.BOX_TARGET
        self.neighbour_dist = config_model.NEIGHBOUR_DIST
        self.giou_conf = giou_conf
        self.ratio_thre = 4.
        self.xywh_mode = config_model.XYWH_MODE
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.n_ch = 5 + self.n_classes
        self.min_anchor_index = min(self.anch_mask)

    def forward(self, xin, labels=None, giou_conf=0.):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin)
        batchsize = output.shape[0]
        fsize = output.shape[2]

        output = output.view(batchsize, self.n_anchors, self.n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

        # logistic activation for xy, obj, cls
        if self.xywh_mode == 'YOLOv3':
            output[..., np.r_[:2, 4:self.n_ch]] = torch.sigmoid(
                output[..., np.r_[:2, 4:self.n_ch]])
        else:
            output = torch.sigmoid(output)

        # calculate pred - xywh obj cls

        x_shift = self.dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = self.dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = self.dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        h_anchors = self.dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

        pred = output.clone()
        if self.xywh_mode == 'YOLOv3':
            pred[..., 0] += x_shift
            pred[..., 1] += y_shift
            pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
            pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors
        else:
            pred[..., 0] = output[..., 0] * (self.neighbour_dist * 2 + 1) - self.neighbour_dist + x_shift
            pred[..., 1] = output[..., 1] * (self.neighbour_dist * 2 + 1) - self.neighbour_dist + y_shift
            pred[..., 2] = torch.pow(output[..., 2], 2) * 4 * w_anchors
            pred[..., 3] = torch.pow(output[..., 3], 2) * 4 * h_anchors

        if self.training is False:
            pred[..., :4] *= self.stride
            return pred.reshape(batchsize, -1, self.n_ch).data

        output_ = pred.clone()
        pred = pred[..., :4].data

        target, obj_mask, tgt_mask, tgt_scale = self.assign_targets(labels, batchsize, fsize, pred)

        if self.giou_loss:
            loss_dict = self.giou_loss(output_, target, tgt_mask, obj_mask)
        else:
            loss_dict = self.yolov3_loss(output, target, obj_mask, tgt_mask, tgt_scale)

        return loss_dict

    def yolov3_loss(self, output, target, obj_mask, tgt_mask, tgt_scale):
        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:self.n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:self.n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale,
                             reduction='sum')  # weighted BCEloss
        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])

        if self.n_classes == 1:
            loss_cls = 0
        else:
            loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])

        loss_dict = {
            'xy': loss_xy,
            'wh': loss_wh,
            'obj': loss_obj,
            'cls': loss_cls,
        }

        return loss_dict

    def giou_loss(self, output_, target, tgt_mask, obj_mask):
        tgt_index = torch.nonzero(tgt_mask[..., :4].reshape(-1, 4)[:, 0])
        if tgt_index.shape[0] > 0:
            loss_giou = giou_loss(
                xywh_to_xyxy((output_[..., :4] * tgt_mask[..., :4]).reshape(-1, 4))[tgt_index],
                xywh_to_xyxy((target[..., :4] * tgt_mask[..., :4]).reshape(-1, 4))[tgt_index],
                reduction='mean'
            )
        else:
            loss_giou = 0.

        output_[..., 4] *= obj_mask
        target[..., 4] *= obj_mask
        loss_obj = self.bce_loss(output_[..., 4], target[..., 4])
        loss = loss_giou * self.giou_weight + loss_obj
        return loss * output_.shape[0], loss_giou, 0, loss_obj, 0, loss

    def assign_targets(self, labels, batchsize, fsize, pred):

        # target assignment

        tgt_mask = torch.zeros(batchsize, self.n_anchors,
                               fsize, fsize, 4 + self.n_classes).type(self.dtype)
        obj_mask = torch.ones(batchsize, self.n_anchors,
                              fsize, fsize).type(self.dtype)
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).type(self.dtype)

        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, self.n_ch).type(self.dtype)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16)
        truth_j_all = truth_y_all.to(torch.int16)

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = self.dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            if self.target_assignment_mode == 'YOLOv3':
                anchor_ious_all = bboxes_iou(truth_box, self.ref_anchors)
                max_anchors = torch.argmax(anchor_ious_all, dim=1)
                ti = torch.where(
                    torch.sum(
                        torch.stack(
                            [max_anchors == m for m in self.anch_mask]
                        )
                        , dim=0)
                )[0]   # how could i modify this line?
                ai = max_anchors[ti] - self.min_anchor_index
            elif self.target_assignment_mode == 'IOU':
                anchor_ious_all = bboxes_iou(truth_box, self.ref_anchors)[:, self.anch_mask]
                anchor_ious_all_over_thre = torch.where(anchor_ious_all > self.iou_thre)
                ti, ai = anchor_ious_all_over_thre[0], anchor_ious_all_over_thre[1]
            elif  self.target_assignment_mode == 'ratio':
                w_ratio = truth_box[:n, 2] / self.ref_anchors[self.anch_mask, 2, None]  # 9, n
                h_ratio = truth_box[:n, 3] / self.ref_anchors[self.anch_mask, 3, None]
                anchor_ious_all_over_thre = torch.where((w_ratio < self.ratio_thre) &
                                                        (1. / w_ratio < self.ratio_thre) &
                                                        (h_ratio < self.ratio_thre) &
                                                        (1. / h_ratio < self.ratio_thre))
                ai, ti = anchor_ious_all_over_thre[0], anchor_ious_all_over_thre[1]


            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(
                pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~pred_best_iou

            if len(ai) == 0:
                continue

            # for ti in range(best_n.shape[0]):
            if self.cross_target:
                gt_i, gt_j, ai, ti = add_neighbours(truth_i[ti],
                                                    truth_x_all[b, ti],
                                                    truth_j[ti],
                                                    truth_y_all[b, ti],
                                                    ai,
                                                    ti,
                                                    fsize,
                                                    d=self.neighbour_dist)
                gt_i, gt_j = gt_i.long(), gt_j.long()
            else:
                gt_i = truth_i[ti].long()
                gt_j = truth_j[ti].long()

            obj_mask[b, ai, gt_j, gt_i] = 1
            tgt_mask[b, ai, gt_j, gt_i, :] = 1
            target[b, ai, gt_j, gt_i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
            target[b, ai, gt_j, gt_i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
            target[b, ai, gt_j, gt_i, 2] = torch.log(
                        truth_w_all[b, ti] / self.dtype(self.masked_anchors)[ai, 0] + 1e-16)
            target[b, ai, gt_j, gt_i, 3] = torch.log(
                        truth_h_all[b, ti] / self.dtype(self.masked_anchors)[ai, 1] + 1e-16)
            target[b, ai, gt_j, gt_i, 4] = 1
            tgt_scale[b, ai, gt_j, gt_i, 0] = torch.sqrt(
                2 - torch.mul(truth_w_all[b, ti], truth_h_all[b, ti]) / fsize / fsize)
            """
            target[b, ai, gt_j, gt_i, 4] = giou(
                xywh_to_xyxy(output[b, ai, gt_j, gt_i, :4]),
                xywh_to_xyxy(target[b, ai, gt_j, gt_i, :4])).detach().clamp(0) * giou_conf + (1 - giou_conf)
            """
        return target, obj_mask, tgt_mask, tgt_scale


class YOLOMultiLayers(nn.Module):
    def __init__(self, config_model, in_channels, cuda=True):
        super(YOLOMultiLayers, self).__init__()
        self.yolo_layer_P5 = YOLOLayer(config_model, layer_no=0, in_ch=in_channels[0], cuda=cuda)
        self.yolo_layer_P4 = YOLOLayer(config_model, layer_no=1, in_ch=in_channels[1], cuda=cuda)
        self.yolo_layer_P3 = YOLOLayer(config_model, layer_no=2, in_ch=in_channels[2], cuda=cuda)

    def forward(self, x, targets=None):
        assert len(x) == 3
        output = list()
        output.append(self.yolo_layer_P5(x[0], targets))
        output.append(self.yolo_layer_P4(x[1], targets))
        output.append(self.yolo_layer_P3(x[2], targets))
        if self.training:
            loss_dict = output[0]
            for o in output[1:]:
                for name, loss in o.items():
                    loss_dict[name] += loss
            #for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'], loss_dict):
            #    self.loss_dict[name] += loss
            return loss_dict
        else:
            return torch.cat(output, dim=1)




def build_yolo_layer(config_model, in_channels, cuda=True):
    return YOLOMultiLayers(config_model, in_channels, cuda=cuda)



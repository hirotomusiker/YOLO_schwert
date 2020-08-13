import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from yolo_schwert.modeling.build import build_model
from yolo_schwert.data.cocodataset import COCODataset
from yolo_schwert.solver import build_lr_scheduler
from yolo_schwert.evaluation import build_evaluator
from yolo_schwert.checkpoint import YOLOCheckpointer


class Trainer:
    def __init__(self, cfg, cuda=False):
        self.device = 'cuda' if cuda else 'cpu'
        self.model = build_model(cfg, cuda).to(self.device)
        self.reduction_mode = cfg.MODEL.LOSS_REDUCTION
        if self.reduction_mode == 'sum':
            # YOLOv3 mode
            self.accum_size = cfg.SOLVER.BATCHSIZE * cfg.SOLVER.SUBDIVISION
        else:
            self.accum_size = 1
        self.dataloader = self.build_train_loader(cfg)
        self.dataiterator = iter(self.dataloader)
        self.optimizer = self.build_optimizer(cfg)
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.subdivision = cfg.SOLVER.SUBDIVISION
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)
        self.log_interval = cfg.LOG_INTERVAL
        self.eval_interval = cfg.TEST.INTERVAL
        self.checkpoint_interval = cfg.CHECKPOINT_INTERVAL
        self.evaluator = self.build_evaluator(cfg)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self._writer = SummaryWriter(cfg.OUTPUT_DIR)
        self.checkpointer = YOLOCheckpointer(
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.random_resize = cfg.AUG.RANDOM_RESIZE

        self.iter_i = 0
        if cfg.MODEL.WEIGHTS:
            checkpoint = self.checkpointer.load(cfg.MODEL.WEIGHTS)
            self.iter_i = checkpoint.get("iteration", -1) + 1



    @classmethod
    def build_train_loader(cls, cfg):
        dataset = COCODataset(cfg, train=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.SOLVER.BATCHSIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            drop_last=True)
        return dataloader

    def build_optimizer(self, cfg):
        base_lr = cfg.SOLVER.LR / self.accum_size
        momentum = cfg.SOLVER.MOMENTUM
        decay = cfg.SOLVER.DECAY
        params_dict = dict(self.model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if 'conv.weight' in key:
                params += [{'params': value, 'weight_decay': decay * self.accum_size}]
            else:
                params += [{'params': value, 'weight_decay': 0.0}]
        optimizer = torch.optim.SGD(params, lr=base_lr, momentum=momentum,
                              dampening=0, weight_decay=decay * self.accum_size)
        return optimizer

    @classmethod
    def build_evaluator(cls, cfg):
        return build_evaluator('COCO', cfg)

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        # start training loop

        for iter_i in range(self.iter_i, self.max_iter):
            self.iter_i = iter_i
            self.optimizer.zero_grad()
            for inner_iter_i in range(self.subdivision):
                try:
                    imgs, targets, _, _ = next(self.dataiterator)  # load a batch
                except StopIteration:
                    self.dataiterator = iter(self.dataloader)
                    imgs, targets, _, _ = next(self.dataiterator)  # load a batch
                imgs = imgs.to(self.device).float()
                targets = targets.to(self.device).float().detach()
                loss_dict = self.model(imgs, targets)
                loss = sum(loss_dict.values())
                assert torch.isfinite(loss).all(), loss_dict
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            loss_dict['total_loss'] = loss
            self.hooks(loss_dict)

    def hooks(self, loss_dict):
        if self.iter_i % self.log_interval == 0:
            loss_dict = {k:v.item() for (k, v) in loss_dict.items()}
            self.log_losses(self.iter_i, loss_dict)
            self.log_tfboard(loss_dict, prefix='train')

        if self.iter_i % self.eval_interval == 0 and self.iter_i > 0:
            eval_results = self.evaluator.evaluate(self.model)
            self.log_tfboard(eval_results, prefix='val')
            self.model.train()

        if self.iter_i % self.checkpoint_interval == 0 and self.iter_i > 0:
            # save the latest checkpoint
            additional_state = {"iteration": self.iter_i}
            self.checkpointer.save('snapshot_latest', **additional_state)

        # random resizing. this will be implemented as a transform module
        if self.iter_i % 10 == 0 and self.iter_i > 0 and self.random_resize:
            imgsize = (np.random.randint(10) % 10 + 10) * 32
            self.dataloader.dataset.img_size = imgsize
            self.dataiterator = iter(self.dataloader)

    def log_losses(self, iter_i, loss_dict):
        # TODO: log smoothed losses
        current_lr = self.scheduler.get_lr()[0] * self.accum_size
        # current_lr = scheduler.get_lr()[0] * batch_size * subdivision
        print('Iter {} / {}, lr: {:.7f}, imagesize: {}, '
              'Losses: xy: {:.3f}, wh: {:.3f}, obj: {:.3f}, cls: {:.3f} total: {:.3f}'.format(
            iter_i, self.max_iter, current_lr,
            self.dataloader.dataset.img_size,
            loss_dict['xy'], loss_dict['wh'],
            loss_dict['obj'], loss_dict['cls'],
            loss_dict['total_loss']),
            flush=True
        )

    def log_tfboard(self, log_dict, prefix=None):
        for k, v in log_dict.items():
            name = prefix + '/' + k
            self._writer.add_scalar(name, v, self.iter_i)

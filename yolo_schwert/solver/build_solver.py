# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from bisect import bisect_right


class WarmUpMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    scheduler module with warm-up phase and lr drop phase
    """
    def __init__(self, optimizer, milestones, gamma, warmup_iters, warmup_power=4, last_epoch=-1):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_iters = warmup_iters
        self.warmup_power = warmup_power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor(self.last_epoch, self.warmup_iters, self.warmup_power)
        lrs = [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
        return lrs


def _get_warmup_factor(i, burn_in, power=4.):
    """
    yolov3-style warm-up schedule
    Args:
        i (int): iteration number
        burn_in (int): iteration number where the warm-up phase ends
        power (float): warm-up factor is calculated as (i / burn_in)^power
    Returns:
        factor (float): warm-up factor
    """
    if i < burn_in:
        factor = pow(i / burn_in, power)
    else:
        factor = 1.
    return factor


def build_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER == 'WarmUpMultiStepLR':
        scheduler = WarmUpMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            gamma=0.1,
            warmup_iters=cfg.SOLVER.BURN_IN
        )
    else:
        raise NotImplementedError()
    return scheduler


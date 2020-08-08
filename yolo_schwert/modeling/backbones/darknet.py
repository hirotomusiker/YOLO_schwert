import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_schwert.modeling.layers import conv, resblock


class DarkNet53(nn.Module):
    """
    DarkNet53 backbone.
    """
    def __init__(self, cfg):
        super(DarkNet53, self).__init__()
        act = 'lrelu'
        self.block_0 = nn.Sequential(
            conv(in_ch=3, out_ch=32, ksize=3, stride=1, act=act),
            conv(in_ch=32, out_ch=64, ksize=3, stride=2, act=act),
            resblock(ch=64, act=act),
            conv(in_ch=64, out_ch=128, ksize=3, stride=2, act=act),
            resblock(ch=128, nblocks=2, act=act),
            conv(in_ch=128, out_ch=256, ksize=3, stride=2, act=act),
            resblock(ch=256, nblocks=8, act=act)
        )

        self.block_1 = nn.Sequential(
            conv(in_ch=256, out_ch=512, ksize=3, stride=2, act=act),
            resblock(ch=512, nblocks=8, act=act)
        )
        self.block_2 = nn.Sequential(
            conv(in_ch=512, out_ch=1024, ksize=3, stride=2, act=act),
            resblock(ch=1024, nblocks=4, act=act)
        )

    def forward(self, x):
        output = []
        x = self.block_0(x)
        output.append(x)
        x = self.block_1(x)
        output.append(x)
        x = self.block_2(x)
        output.append(x)
        return output


def build_darknet(cfg):
    if cfg.BACKBONE == "darknet53":
        return DarkNet53(cfg)
    else:
        raise NotImplementedError()





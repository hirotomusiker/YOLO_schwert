import torch
from torch import nn
from yolo_schwert.modeling.layers import conv, resblock, SPPModule


class YOLOv3Neck(nn.Module):
    """
    YOLOv3 Neck that consists of FPN-like up-sampling layers and branches
    """
    def __init__(self, config_model, in_channels):
        """
        Args:
            config_model: config.MODEL
            in_channels (list of int): channel numbers of input and two shortcuts
        """
        super(YOLOv3Neck, self).__init__()
        act = config_model.ACT
        #if use_spp:
        #   self_block_1 = SPPModule(in_ch=512, act=act)
        #  F.interpolate(prev_features, scale_factor=2, mode="nearest")

        self.block_0 = nn.Sequential(
            resblock(ch=in_channels[0], nblocks=2, shortcut=False, act=act),
            conv(in_ch=in_channels[0], out_ch=512, ksize=1, stride=1, act=act),
        )
        self.branch_P5 = conv(in_ch=512, out_ch=1024, ksize=3, stride=1, act=act)
        self.block_1 = nn.Sequential(
            conv(in_ch=512, out_ch=256, ksize=1, stride=1, act=act),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.block_2 = nn.Sequential(
            conv(in_ch=256 + in_channels[1], out_ch=256, ksize=1, stride=1, act=act),
            conv(in_ch=256, out_ch=512, ksize=3, stride=1, act=act),
            resblock(ch=512, nblocks=1, shortcut=False, act=act),
            conv(in_ch=512, out_ch=256, ksize=1, stride=1, act=act)
        )
        self.branch_P4 = conv(in_ch=256, out_ch=512, ksize=3, stride=1, act=act)
        self.block_3 = nn.Sequential(
            conv(in_ch=256, out_ch=128, ksize=1, stride=1, act=act),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.block_4 = nn.Sequential(
            conv(in_ch=128 + in_channels[2], out_ch=128, ksize=1, stride=1, act=act),
            conv(in_ch=128, out_ch=256, ksize=3, stride=1, act=act),
            resblock(ch=256, nblocks=2, shortcut=False, act=act)
        )

    def forward(self, x):
        output = list()
        assert len(x) == 3
        h = self.block_0(x[2])
        output.append(self.branch_P5(h))
        h = self.block_1(h)
        h = torch.cat((h, x[1]), dim=1)
        h = self.block_2(h)
        output.append(self.branch_P4(h))
        h = self.block_3(h)
        h = torch.cat((h, x[0]), dim=1)
        output.append(self.block_4(h))
        return output


def build_yolov3_neck(config_model, in_channels):
    return YOLOv3Neck(config_model, in_channels)

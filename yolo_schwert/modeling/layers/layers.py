from torch import nn


def conv(in_ch, out_ch, ksize, stride, act='lrelu'):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if act == 'lrelu':
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    elif act == 'mish':
        stage.add_module('mish', Mish())
    return stage


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nblocks=1, shortcut=True, act='lrelu'):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(conv(ch, ch//2, 1, 1, act=act))
            resblock_one.append(conv(ch//2, ch, 3, 1, act=act))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

class SPPModule(nn.Module):
    """DC-SPP-YOLO arxiv.org/abs/1903.08589
    """
    def __init__(self, in_ch, act='lrelu'):
        super(SPPModule, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5,  stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9,  stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.out_conv = conv(in_ch=in_ch*4, out_ch=in_ch*2, ksize=1, stride=1, act=act)

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        x = torch.cat([x, p1, p2, p3], dim=1)
        x = self.out_conv(x)
        return x

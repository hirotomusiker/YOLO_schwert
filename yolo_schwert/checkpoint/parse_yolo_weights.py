from __future__ import division
import torch
import numpy as np


def parse_conv_block(m, weights, offset, initflag):
    """
    Initialization of conv layers with batchnorm
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    conv_model = m[0]
    bn_model = m[1]
    param_length = m[1].bias.numel()
    # batchnorm
    for pname in ['bias', 'weight', 'running_mean', 'running_var']:
        layerparam = getattr(bn_model, pname)

        if initflag:  # yolo initialization - scale to one, bias to zero
            if pname == 'weight':
                weights = np.append(weights, np.ones(param_length))
            else:
                weights = np.append(weights, np.zeros(param_length))

        param = torch.from_numpy(weights[offset:offset + param_length]).view_as(layerparam)
        layerparam.data.copy_(param)
        offset += param_length

    param_length = conv_model.weight.numel()

    # conv
    if initflag:  # yolo initialization
        n, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.weight)
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights


def parse_yolo_block(m, weights, offset, initflag):
    """
    YOLO Layer (one conv with bias) Initialization
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    conv_model = m._modules['conv']
    param_length = conv_model.bias.numel()

    if initflag:  # yolo initialization - bias to zero
        weights = np.append(weights, np.zeros(param_length))

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.bias)
    conv_model.bias.data.copy_(param)
    offset += param_length

    param_length = conv_model.weight.numel()

    if initflag:  # yolo initialization
        n, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.weight)
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights


def parse_modules(m, weights, offset, initflag=False):
    if m._get_name() == 'Sequential':
        # normal conv block
        offset, weights = parse_conv_block(m, weights, offset, initflag)
    elif m._get_name() == 'resblock':
        # residual block
        for modu in m._modules['module_list']:
            for blk in modu:
                offset, weights = parse_conv_block(blk, weights, offset, initflag)
    # print(m._get_name() , offset, len(weights))
    return offset, weights


def parse_yolo_weights(model, weights_path, initflag=False):
    """
    Parse darknet / yolo weights on the model.
    When parsing darknet weights, the neck / dense head layers are initialized.
    Args:
        model: pytorch model object
        weights_path (str): weights file path
        initflag (bool): initialize the non-backbone layers or not
    Returns:

    """

    fp = open(weights_path, "rb")
    # skip the header
    _ = np.fromfile(fp, dtype=np.int32, count=5)  # not used
    # read weights
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    offset = 0
    for m in model.backbone.block_0:
        offset, weights = parse_modules(m, weights, offset)
    for m in model.backbone.block_1:
        offset, weights = parse_modules(m, weights, offset)
    for m in model.backbone.block_2:
        offset, weights = parse_modules(m, weights, offset)
    for m in model.neck.block_0:
        offset, weights = parse_modules(m, weights, offset, initflag=initflag)
    offset, weights = parse_conv_block(model.neck.branch_P5, weights, offset, initflag=initflag)
    offset, weights = parse_yolo_block(model.dense_head.yolo_layer_P5, weights, offset, initflag=initflag)
    for m in model.neck.block_1:
        offset, weights = parse_modules(m, weights, offset, initflag=initflag)
    for m in model.neck.block_2:
        offset, weights = parse_modules(m, weights, offset, initflag=initflag)
    offset, weights = parse_conv_block(model.neck.branch_P4, weights, offset, initflag=initflag)
    offset, weights = parse_yolo_block(model.dense_head.yolo_layer_P4, weights, offset, initflag=initflag)
    for m in model.neck.block_3:
        offset, weights = parse_modules(m, weights, offset, initflag=initflag)
    for m in model.neck.block_4:
        offset, weights = parse_modules(m, weights, offset, initflag=initflag)
    offset, weights = parse_yolo_block(model.dense_head.yolo_layer_P3, weights, offset, initflag=initflag)
    print("model {} has been loaded: {} / {}".format(weights_path, offset, len(weights)))
    return {"model": model.state_dict()}




class EfficientYOLOv3(nn.Module):
    # TODO: refactor this using backbone / neck / head split
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self, config_model, ignore_thre=0.7, giou_conf=0.):
        """
        Initialization of YOLOv3 class.
        Args:
            config_model (dict): used in YOLOLayer.
            ignore_thre (float): used in YOLOLayer.
        """
        super(EfficientYOLOv3, self).__init__()

        if config_model['BACKBONE'] == 'efficientnet':
            self.backbone = build_efficientnet_backbone(config_model)
            out_features = [v for (k, v) in self.backbone._out_feature_channels.items()]
            out_features = out_features[1:]
        else:
            raise Exception('Model name {} is not available'.format(config_model['BACKBONE']))
        self.head = build_yolov3_head(config_model,
                              ignore_thre,
                              use_spp=config_model['SPP'],
                              act=config_model['ACT'],
                              in_channels = out_features,
                              giou_conf=giou_conf)
        self.effnet = config_model['BACKBONE'] == 'efficientnet'
        self.giou_conf = giou_conf

        if False:
            for m in model.head:
                if m._get_name() == 'YOLOLayer':
                    torch.nn.init.constant_(m._modules['conv'].bias, 0)

    def forward(self, x, targets=None):
        train = targets is not None
        output = []
        route_layers = []
        self.loss_dict = defaultdict(float)
        features = self.backbone(x)
        if self.effnet:
            x = features['eff5']
            # eff2, 3, 4 : torch.Size([2, 32, 256, 256]) torch.Size([2, 56, 128, 128]) torch.Size([2, 160, 64, 64])
            route_layers.append(features['eff3'])
            route_layers.append(features['eff4'])
        for i, module in enumerate(self.head):
            if i in [3, 11, 17]:
                if train:
                    x, *loss_dict = module(x, targets, self.giou_conf)
                    for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'] , loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)
                output.append(x)
            else:
                x = module(x)
            # route layers
            if i in [1, 9]:
                route_layers.append(x)
            if i == 3:
                x = route_layers[2]
            if i == 11:  # yolo 2nd
                x = route_layers[3]
            if i == 5:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 13:
                x = torch.cat((x, route_layers[0]), 1)

        if train:
            return sum(output)
        else:
            return torch.cat(output, 1)

from collections import defaultdict

import torch.nn as nn

from utils.utils import *


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        dilation=1,
                                                        padding=pad,
                                                        bias=not bn))

            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='bilinear', align_corners=True)
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [float(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs)
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, nC, img_dim, anchor_idxs):
        super(YOLOLayer, self).__init__()

        mat = scipy.io.loadmat('utils/targets_60c.mat')
        wh = mat['class_stats'][:, 6:].reshape(-1, 3)
        area = wh[:, 1] * wh[:, 2]
        i = np.argsort(area)  # least to greatest

        nA = int(len(i) / 3)  # anchors per yolo layer
        if anchor_idxs[0] == 40:  # 6
            stride = 32
            i = i[nA * 2:nA * 3]
        elif anchor_idxs[0] == 20:  # 3
            stride = 16
            i = i[nA:nA * 2]
        else:
            stride = 8
            i = i[:nA]

        classes = wh[i, 0]
        wh = wh[i, 1:]
        anchors = [(a_w, a_h) for a_w, a_h in wh]  # (pixels)
        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (60)
        self.bbox_attrs = 5
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        # Build anchor grids
        nG = int(self.img_dim / stride)
        nB = 1  # batch_size set to 1
        shape = [nB, self.nA, nG, nG]
        self.grid_x = torch.arange(nG).repeat(nG, 1).repeat(nB * nA, 1, 1).view(shape).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().repeat(nB * nA, 1, 1).view(shape).float()
        self.scaled_anchors = torch.FloatTensor(wh / stride)
        self.anchor_w = self.scaled_anchors[:, 0:1].repeat(nB, 1).repeat(1, 1, nG * nG).view(shape)
        self.anchor_h = self.scaled_anchors[:, 1:2].repeat(nB, 1).repeat(1, 1, nG * nG).view(shape)
        self.anchor_wh = torch.cat((self.anchor_w.unsqueeze(4), self.anchor_h.unsqueeze(4)), 4).squeeze(0)
        self.anchor_class = torch.FloatTensor(classes).view(-1, 1).repeat(nB, 1).repeat(1, 1, nG * nG).view(shape)

        self.Sigmoid = torch.nn.Sigmoid()
        self.Tanh = torch.nn.Tanh()

    def forward(self, p, targets=None, requestPrecision=False):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
        device = torch.device('cuda:0' if p.is_cuda else 'cpu')
        weight = xview_class_weights(range(60)).to(device)
        bs = p.shape[0]
        nG = p.shape[2]
        stride = self.img_dim / nG

        BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)
        MSELoss = nn.MSELoss(reduce=False)
        # CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight)

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x = self.grid_x.cuda()
            self.grid_y = self.grid_y.cuda()
            self.anchor_w = self.anchor_w.cuda()
            self.anchor_h = self.anchor_h.cuda()

        # x.view(4, 650, 19, 19) -- > (4, 10, 19, 19, 65)  # (bs, anchors, grid, grid, classes + xywh) OLD
        # x.view(4, 100, 19, 19) -- > (4, 20, 19, 19, 5)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Get outputs
        x = self.Sigmoid(p[..., 0])  # Center x
        y = self.Sigmoid(p[..., 1])  # Center y
        w = self.Sigmoid(p[..., 2])  # Width
        h = self.Sigmoid(p[..., 3])  # Height
        width = ((w.data * 2) ** 2) * self.anchor_w
        height = ((h.data * 2) ** 2) * self.anchor_h

        # Add offset and scale with anchors (in grid space, i.e. 0-13)
        pred_boxes = FT(p[..., :4].shape)
        pred_conf = p[..., 4]  # Conf
        pred_cls = self.anchor_class.repeat(bs, 1, 1, 1)

        # Training
        if targets is not None:
            if requestPrecision:
                pred_boxes[..., 0] = x.data + self.grid_x - width / 2
                pred_boxes[..., 1] = y.data + self.grid_y - height / 2
                pred_boxes[..., 2] = x.data + self.grid_x + width / 2
                pred_boxes[..., 3] = y.data + self.grid_y + height / 2

            tx, ty, tw, th, mask, tcls, TP, FP, FN, TC, ap, good_anchors = \
                build_targets_new(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nG,
                                  self.anchor_wh, requestPrecision)

            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()
                good_anchors = good_anchors.cuda()

            # Mask outputs to ignore non-existing objects (but keep confidence predictions)
            nM = mask.sum().float()
            nGT = FT([sum([len(x) for x in targets])])
            if nM > 0:
                # wA = nM / nGT  # weight anchor-grid
                wC = weight[torch.argmax(tcls, 1)]  # weight class
                wC /= sum(wC)
                lx = 5 * (MSELoss(x[mask], tx[mask]) * wC).sum()
                ly = 5 * (MSELoss(y[mask], ty[mask]) * wC).sum()
                lw = 5 * (MSELoss(w[mask], tw[mask]) * wC).sum()
                lh = 5 * (MSELoss(h[mask], th[mask]) * wC).sum()
                lconf = (BCEWithLogitsLoss(pred_conf[mask], mask[mask].float()) * wC).sum()
                lcls = FT([0])
            else:
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])
                # wA = FT([1])

            lconf += 0.5 * BCEWithLogitsLoss(pred_conf[~good_anchors], good_anchors[~good_anchors].float()).mean()

            loss = lx + ly + lw + lh + lconf
            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                   ap, nGT, TP, FP, FN, TC, 0, 0

        else:
            pred_boxes[..., 0] = x.data + self.grid_x
            pred_boxes[..., 1] = y.data + self.grid_y
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            # If not in training phase return predictions
            output = torch.cat(
                (pred_boxes.view(bs, -1, 4) * stride, pred_conf.view(bs, -1, 1), pred_cls.view(bs, -1, 1)), -1)
            return output.data


class YOLOLayerOld(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, nC, img_dim, anchor_idxs):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (60)
        self.bbox_attrs = 5 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        # Build anchor grids
        nG = int(self.img_dim / stride)
        nB = 1  # batch_size set to 1
        shape = [nB, self.nA, nG, nG]
        self.grid_x = torch.arange(nG).repeat(nG, 1).repeat(nB * nA, 1, 1).view(shape).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().repeat(nB * nA, 1, 1).view(shape).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].repeat(nB, 1).repeat(1, 1, nG * nG).view(shape)
        self.anchor_h = self.scaled_anchors[:, 1:2].repeat(nB, 1).repeat(1, 1, nG * nG).view(shape)
        self.anchor_wh = torch.cat((self.anchor_w.unsqueeze(4), self.anchor_h.unsqueeze(4)), 4).squeeze(0)
        self.Sigmoid = torch.nn.Sigmoid()
        self.Tanh = torch.nn.Tanh()

    # @profile
    def forward(self, p, targets=None, requestPrecision=False):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
        device = torch.device('cuda:0' if p.is_cuda else 'cpu')
        weight = xview_class_weights(range(60)).to(device)

        bs = p.shape[0]
        nG = p.shape[2]
        stride = self.img_dim / nG

        BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)
        MSELoss = nn.MSELoss(reduce=True)
        CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight)

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x = self.grid_x.cuda()
            self.grid_y = self.grid_y.cuda()
            self.anchor_w = self.anchor_w.cuda()
            self.anchor_h = self.anchor_h.cuda()

        # x.view(4, 650, 19, 19) -- > (4, 10, 19, 19, 65)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Get outputs
        x = self.Sigmoid(p[..., 0])  # Center x
        y = self.Sigmoid(p[..., 1])  # Center y
        w = self.Sigmoid(p[..., 2])  # Width
        h = self.Sigmoid(p[..., 3])  # Height
        width = w.data * 2 * self.anchor_w
        height = h.data * 2 * self.anchor_h


        # Add offset and scale with anchors (in grid space, i.e. 0-13)
        pred_boxes = FT(p[..., :4].shape)
        pred_conf = p[..., 4]  # Conf
        pred_cls = p[..., 5:]  # Class

        # Training
        if targets is not None:
            if requestPrecision:
                pred_boxes[..., 0] = x.data + self.grid_x - width / 2
                pred_boxes[..., 1] = y.data + self.grid_y - height / 2
                pred_boxes[..., 2] = x.data + self.grid_x + width / 2
                pred_boxes[..., 3] = y.data + self.grid_y + height / 2

            tx, ty, tw, th, mask, tcls, TP, FP, FN, TC, ap, good_anchors = \
                build_targets(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nG,
                              self.anchor_wh, requestPrecision)

            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()
                good_anchors = good_anchors.cuda()

            # Mask outputs to ignore non-existing objects (but keep confidence predictions)
            nM = mask.sum().float()
            nGT = FT([sum([len(x) for x in targets])])
            if nM > 0:
                wA = nM / nGT  # weight anchor-grid
                wC = weight[torch.argmax(tcls, 1)]  # weight class
                wC /= sum(wC)
                lx = 5 * wA * MSELoss(x[mask], tx[mask])
                ly = 5 * wA * MSELoss(y[mask], ty[mask])
                lw = 5 * wA * MSELoss(w[mask], tw[mask])
                lh = 5 * wA * MSELoss(h[mask], th[mask])
                lconf = wA * (BCEWithLogitsLoss(pred_conf[mask], mask[mask].float()) * wC).sum()
                lcls = 0.2 * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1)) * wA
            else:
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])
                wA = FT([1])

            lconf += 0.5 * wA * BCEWithLogitsLoss(pred_conf[~good_anchors], good_anchors[~good_anchors].float()).mean()

            loss = lx + ly + lw + lh + lconf + lcls
            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                   ap, nGT, TP, FP, FN, TC, 0, 0

        else:
            pred_boxes[..., 0] = x.data + self.grid_x
            pred_boxes[..., 1] = y.data + self.grid_y
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            # self.nC = 60
            # pred_cls = torch.zeros((bs, self.nA, nG, nG, self.nC)).cuda()
            # pred_cls[:,:,:,:,0]=1

            # If not in training phase return predictions
            output = torch.cat(
                (pred_boxes.view(bs, -1, 4) * stride, pred_conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.nC)), -1)
            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'AP', 'nGT', 'TP', 'FP', 'FN', 'TC', 'precision',
                           'recall']

    def forward(self, x, targets=None, requestPrecision=False):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets, requestPrecision)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            # print(x.shape, module_def['type'])
            layer_outputs.append(x)

        if is_training:
            self.losses['nGT'] /= 3
            self.losses['TC'] /= 3

            ui = np.unique(self.losses['TC'])[1:]
            for i in ui:
                j = self.losses['TC'] == float(i)
                TP = (self.losses['TP'][j] > 0).sum().float()
                FP = (self.losses['FP'][j] > 0).sum().float()
                FN = (self.losses['FN'][j] == 3).sum().float()
                self.losses['precision'] += (TP / (TP + FP + 1e-16)) / len(ui)
                self.losses['recall'] += (TP / (TP + FN + 1e-16)) / len(ui)

            self.losses['TP'] = (self.losses['TP'] > 0).sum().float()
            self.losses['FP'] = (self.losses['FP'] > 0).sum().float()
            self.losses['FN'] = (self.losses['FN'] == 3).sum().float()
            self.losses['TC'] = 0

        return sum(output) if is_training else torch.cat(output, 1)


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

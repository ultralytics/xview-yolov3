from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from utils.parse_config import *
from utils.utils import build_targets
from utils.utils import xview_class_weights


# @profile
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
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

        anchors = [(a_w * img_dim, a_h * img_dim) for a_w, a_h in anchors]
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (60)
        self.bbox_attrs = 5 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser
        self.ignore_thres = 0.5
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        class_weights = xview_class_weights(torch.arange(nC))
        class_weights = class_weights / class_weights.mean()
        self.mse_loss = nn.MSELoss(size_average=True)
        self.bce_loss = nn.BCELoss(size_average=True)
        self.bce_loss_cls = nn.BCELoss(size_average=True, weight=class_weights)

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
        self.scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        anchor_w = self.scaled_anchors.index_select(1, LongTensor([0]))
        anchor_h = self.scaled_anchors.index_select(1, LongTensor([1]))
        self.anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nG * nG).view(shape)
        self.anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nG * nG).view(shape)
        self.anchor_wh = torch.cat((self.anchor_w.unsqueeze(4), self.anchor_h.unsqueeze(4)), 4).squeeze()
        self.nGtotal = (self.img_dim / 32) ** 2 + (self.img_dim / 16) ** 2 + (self.img_dim / 8) ** 2

    #@profile
    def forward(self, x, targets=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        bs = x.shape[0]
        nG = x.shape[2]
        stride = self.img_dim / nG

        # x.view(4, 3, 67, 13, 13) -- > (4, 3, 13, 13, 67)
        prediction = x.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0]) # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = torch.sigmoid(prediction[..., 2])  # Width
        h = torch.sigmoid(prediction[..., 3]) # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        if x.is_cuda and not self.grid_x.is_cuda:
            device = torch.device('cuda:0' if x.is_cuda else 'cpu')
            self.grid_x = self.grid_x.cuda()
            self.grid_y = self.grid_y.cuda()
            self.anchor_w = self.anchor_w.cuda()
            self.anchor_h = self.anchor_h.cuda()
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()

        # Add offset and scale with anchors (in grid space, i.e. 0-13)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = (x.data + self.grid_x) - w.data * self.anchor_w * 5 / 2
        pred_boxes[..., 1] = (y.data + self.grid_y) - h.data * self.anchor_h * 5 / 2
        pred_boxes[..., 2] = (x.data + self.grid_x) + w.data * self.anchor_w * 5 / 2
        pred_boxes[..., 3] = (y.data + self.grid_y) + h.data * self.anchor_h * 5 / 2

        # Training
        if targets is not None:
            tx, ty, tw, th, mask, tcls, TP, FP, FN, nGT, ap = build_targets(pred_boxes,
                                                                            pred_conf,
                                                                            pred_cls,
                                                                            targets,
                                                                            self.scaled_anchors,
                                                                            self.nA,
                                                                            self.nC,
                                                                            nG,
                                                                            self.anchor_wh)

            tcls = tcls[mask]
            if x.is_cuda:
                mask = mask.cuda()
                tx = tx.cuda()
                ty = ty.cuda()
                tw = tw.cuda()
                th = th.cuda()
                tcls = tcls.cuda()

            # Mask outputs to ignore non-existing objects (but keep confidence predictions)
            #nT = FloatTensor([sum([len(x) for x in targets])])
            #weight = mask.sum().float()/nT  # weigh by fraction of targets found in each of the 3 yolo layers
            weight = 1
            if nGT > 0:
                loss_x = 5 * self.mse_loss(x[mask], tx[mask]) * weight
                loss_y = 5 * self.mse_loss(y[mask], ty[mask]) * weight
                loss_w = 5 * self.mse_loss(w[mask], tw[mask]) * weight
                loss_h = 5 * self.mse_loss(h[mask], th[mask]) * weight
                loss_cls = self.bce_loss(pred_cls[mask], tcls.float()) * weight
                loss_conf = self.bce_loss(pred_conf[mask], mask[mask].float()) * weight
            else:
                loss_x, loss_y, loss_w, loss_h, loss_cls, loss_conf = 0, 0, 0, 0, 0, 0

            loss_conf += 0.5 * self.bce_loss(pred_conf[~mask], mask[~mask].float()) * weight

            loss = (loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls)
            return loss, loss.item(), loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), \
                   ap, nGT, TP, FP, FN, 0, 0

        else:
            pred_boxes[..., 0] = x.data + self.grid_x
            pred_boxes[..., 1] = y.data + self.grid_y
            pred_boxes[..., 2] = w.data * self.anchor_w * 5
            pred_boxes[..., 3] = h.data * self.anchor_h * 5
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
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'AP', 'nGT', 'TP', 'FP', 'FN', 'precision',
                           'recall']

    # @profile
    def forward(self, x, targets=None):
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
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            TP = (self.losses['TP'] > 0).float().sum()
            FP = (self.losses['FP'] > 0).float().sum()
            FN = (self.losses['FN'] == 3).float().sum()
            self.losses['precision'] = TP / (TP + FP + 1e-16)
            self.losses['recall'] = TP / (TP + FN + 1e-16)
            self.losses['TP'], self.losses['FP'], self.losses['FN'] = TP, FP, FN

        return sum(output) if is_training else torch.cat(output, 1)

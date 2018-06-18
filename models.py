from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from utils.parse_config import *
from utils.utils import build_targets, xview_class_weights


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
            upsample = nn.Upsample(scale_factor=int(module_def['stride']),
                                   mode='nearest')
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
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs, hyperparams['batch_size'])
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

    def __init__(self, anchors, num_classes, img_dim, anchor_idxs, batch_size=1):
        super(YOLOLayer, self).__init__()
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

        anchors = [(a_w * img_dim, a_h * img_dim) for a_w, a_h in anchors]

        nA = len(anchors)
        self.anchors = anchors
        self.num_anchors = nA
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser
        self.ignore_thres = 0.5
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        self.mse_loss = nn.MSELoss()

        class_weights = 1 / xview_class_weights(torch.arange(num_classes))
        self.bce_loss_cls = nn.BCELoss(weight=class_weights)
        self.bce_loss = nn.BCELoss()

        if anchor_idxs[0] == 6:
            stride = 32
        elif anchor_idxs[0] == 3:
            stride = 16
        else:
            stride = 8

        # Build anchor grids
        g_dim = int(self.img_dim / stride)
        nB = 1  # batch_size set to 1
        shape = [nB, self.num_anchors, g_dim, g_dim]
        self.grid_x = torch.arange(g_dim).repeat(g_dim, 1).repeat(nB * nA, 1, 1).view(shape).type(FloatTensor)
        self.grid_y = torch.arange(g_dim).repeat(g_dim, 1).t().repeat(nB * nA, 1, 1).view(shape).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = self.scaled_anchors.index_select(1, LongTensor([0]))
        anchor_h = self.scaled_anchors.index_select(1, LongTensor([1]))
        self.anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, g_dim * g_dim).view(shape)
        self.anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, g_dim * g_dim).view(shape)
        self.anchor_xywh = torch.cat((self.grid_x.unsqueeze(4), self.grid_y.unsqueeze(4), self.anchor_w.unsqueeze(4),
                                      self.anchor_h.unsqueeze(4)), 4)

        # prepopulate target-class zero matrices for speed
        nB = batch_size
        self.tcls_zeros = torch.zeros(nB, nA, g_dim, g_dim, num_classes).float()  # predefined for 4 batch-size

    # @profile
    def forward(self, x, targets=None, current_img_path=None):
        bs = x.shape[0]
        g_dim = x.shape[2]
        stride = self.img_dim / g_dim
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        prediction = x.view(bs, self.num_anchors, self.bbox_attrs, g_dim, g_dim).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        if x.is_cuda and not self.grid_x.is_cuda:
            self.grid_x = self.grid_x.cuda()
            self.grid_y = self.grid_y.cuda()
            self.anchor_w = self.anchor_w.cuda()
            self.anchor_h = self.anchor_h.cuda()
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.anchor_xywh = self.anchor_xywh.cuda()

        # Add offset and scale with anchors (in grid space, i.e. 0-13)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        # Training
        if targets is not None:
            nGT, ap, tx, ty, tw, th, mask, tcls = build_targets(pred_boxes,
                                                                conf.data.cpu(),
                                                                pred_cls.data.cpu(),
                                                                targets.data.cpu(),
                                                                self.scaled_anchors,
                                                                self.num_anchors,
                                                                self.num_classes,
                                                                g_dim,
                                                                self.tcls_zeros,
                                                                self.anchor_xywh)

            tcls = tcls[mask]
            if x.is_cuda:
                mask = mask.cuda()
                tx = tx.cuda()
                ty = ty.cuda()
                tw = tw.cuda()
                th = th.cuda()
                tcls = tcls.cuda()

            # Mask outputs to ignore non-existing objects (but keep confidence predictions)
            if nGT > 0:
                loss_x = self.lambda_coord * self.mse_loss(x[mask], tx[mask])
                loss_y = self.lambda_coord * self.mse_loss(y[mask], ty[mask])
                loss_w = self.lambda_coord * self.mse_loss(w[mask], tw[mask])
                loss_h = self.lambda_coord * self.mse_loss(h[mask], th[mask])
                loss_cls = self.bce_loss_cls(pred_cls[mask], tcls) * 4
                loss_conf = self.bce_loss(conf[mask], mask[mask].float()) * 8
            else:
                loss_x, loss_y, loss_w, loss_h, loss_cls, loss_conf = 0, 0, 0, 0, 0, 0

            loss_conf += self.lambda_noobj * self.bce_loss(conf[~mask], mask[~mask].float())

            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return loss, loss.item(), loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), \
                   ap, nGT

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (pred_boxes.view(bs, -1, 4) * stride, conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)),
                -1)
            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416, batch_size=1):
        super(Darknet, self).__init__()
        self.current_img_path = ''
        self.module_defs = parse_model_config(config_path)
        self.module_defs[0]['height'] = img_size
        self.module_defs[0]['batch_size'] = batch_size

        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'AP', 'nGT']

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
                    x, *losses = module[0](x, targets, self.current_img_path)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nGT'] /= 3
            self.losses['AP'] /= 3

        return sum(output) if is_training else torch.cat(output, 1)

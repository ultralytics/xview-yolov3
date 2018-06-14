import random

import cv2
import numpy as np
import torch

# set printoptions
torch.set_printoptions(linewidth=320, precision=5)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{11.5g}'.format})  # format short g, %precision=5


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def plot_one_box(x, im, color=None, label=None):
    tl = round(0.003 * max(im.shape[0:2]))  # line thickness
    tf = max(tl - 1, 2)  # font thickness

    c1 = (int(x[0]), int(x[1]))
    c2 = (int(x[2]), int(x[3]))
    cls = int(x[-1])
    if label is None:
        classes = load_classes('data/coco.names')
        label = '{0}'.format(classes[cls])
    color = color or [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    cv2.rectangle(im, c1, c2, color, thickness=tl)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(im, c1, c2, color, -1)  # filled
    cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] > conf_thres]
        # If none are remaining => process next image
        nP = image_pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        box_corner = image_pred.new(nP, 4)
        xy = image_pred[:, 0:2]
        wh = image_pred[:, 2:4] / 2
        box_corner[:, 0:2] = xy - wh
        box_corner[:, 2:4] = xy + wh
        image_pred[:, :4] = box_corner

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.shape[0]:
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


# @profile
def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, dim, ignore_thres,
                  img_dim):
    """
    returns nGT, nCorrect, tx, ty, tw, th, tconf, tcls
    """

    nB = target.shape[0]
    nA = num_anchors
    tx = torch.zeros(nB, nA, dim, dim)
    ty = torch.zeros(nB, nA, dim, dim)
    tw = torch.zeros(nB, nA, dim, dim)
    th = torch.zeros(nB, nA, dim, dim)
    tconf = torch.zeros(nB, nA, dim, dim)
    tcls = torch.zeros(nB, nA, dim, dim, num_classes)

    anchors = torch.FloatTensor(anchors)

    nTP = 0
    nGT = 0
    #nCorrect = 0
    precision, recall = [], []
    for b in range(nB):
        nT = torch.argmin(target[b, :, 4])  # number of targets (measures index of first zero-height target box)
        nGT += nT.item()

        # Convert to position relative to box
        gx, gy, gw, gh = target[b, :nT, 1] * dim, target[b, :nT, 2] * dim, target[b, :nT, 3] * dim, target[b, :nT,
                                                                                                    4] * dim
        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        gi = torch.clamp(gx.long(), max=dim - 1)
        gj = torch.clamp(gy.long(), max=dim - 1)
        # Calculate ious between ground truth and each of the 3 anchors
        iou = []
        for i in range(nA):
            iou.append(bbox_iou(target[b, :nT, 1:] * dim, pred_boxes[b, i, gj, gi], x1y1x2y2=False))
        iou = torch.cat(iou).view(nA, -1)
        # Select best iou and anchor
        iou, best_a = torch.max(iou, 0)

        # Coordinates
        tx[b, best_a, gj, gi] = gx - gi.float()
        ty[b, best_a, gj, gi] = gy - gj.float()
        # Width and height
        tw[b, best_a, gj, gi] = torch.log(gw / anchors[best_a, 0] + 1e-16)
        th[b, best_a, gj, gi] = torch.log(gh / anchors[best_a, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_a, gj, gi, target[b, :nT, 0].long()] = 1
        tconf[b, best_a, gj, gi] = 1
        # predicted classes and confidence
        pcls = torch.argmax(pred_cls[b, best_a, gj, gi], 1)
        pconf = pred_conf[b, best_a, gj, gi]

        TP = ((iou > 0.5) & (pconf > 0.5) & (pcls.float() == target[b, :nT, 0])).float()
        FP = ((iou > 0.5) & (pconf > 0.5) & (pcls.float() != target[b, :nT, 0])).float()
        FN = ((iou < 0.5) | (pconf < 0.5)).float()

        precision.extend((TP / (TP + FP + 1e-16)).tolist())
        recall.extend((TP / (TP + FN + 1e-16)).tolist())
        #nCorrect += TP.sum().item()

        nTP += TP.sum().item()

    ap = nTP/nGT #compute_ap(recall, precision)
    return nGT, ap, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='uint8')[y])

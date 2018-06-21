import random

import cv2
import numpy as np
import torch

# set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='short')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{11.5g}'.format})  # format short g, %precision=5


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def xview_indices2classes(indices):  # remap xview classes 11-94 to 0-61
    class_list = [11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 40, 41,
                  42, 44, 45, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 71, 72, 73, 74,
                  76, 77, 79, 83, 84, 86, 89, 91, 93, 94]
    return class_list[indices]


def xview_class_weights(indices):  # weights of each class in the training set, normalized to mu = 1
    weights = torch.FloatTensor(
        [0.0074, 0.0367, 0.0716, 0.0071, 0.295, 21.1, 0.695, 0.11, 0.363, 1.22, 0.588, 0.364, 0.0859, 0.409, 0.0894,
         0.0149, 0.0173, 0.0017, 0.163, 0.184, 0.0125, 0.0122, 0.0124, 0.0687, 0.146, 0.0701, 0.0226, 0.0191, 0.0797,
         0.0202, 0.0449, 0.0331, 0.0083, 0.0204, 0.0156, 0.0193, 0.007, 0.0064, 0.0337, 0.135, 0.0337, 0.0078, 0.0628,
         0.0843, 0.0286, 0.0083, 0.071, 0.119, 31.6, 0.0208, 0.109, 0.0949, 0.122, 0.425, 0.0125, 0.171, 0.237, 0.158,
         0.0373, 0.0085])
    return weights[indices.long()]


def plot_one_box(x, im, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.003 * max(im.shape[0:2]))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 2)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


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
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


#@profile
def build_targets(pred_boxes, pred_conf, pred_cls, target, anchor_wh, nA, nC, nG, anchor_xywh, anchor_a):
    """
    returns nGT, nCorrect, tx, ty, tw, th, tconf, tcls
    """

    nB = target.shape[0]
    tx = torch.zeros(nB, nA, nG, nG)  # batch size (4), number of anchors (3), number of grid points (13)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.zeros(nB, nA, nG, nG)
    #tcls = torch.zeros(nB, nA, nG, nG, nC)  # nC = number of classes
    tcls = torch.cuda.FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    precision, recall, nGT = [], [], 0
    TP = torch.zeros(nB, 7607)
    FP = torch.zeros(nB, 7607)
    FN = torch.zeros(nB, 7607)
    for b in range(nB):
        nT = torch.argmin(target[b, :, 4]).item()  # number of targets (measures index of first zero-height target box)
        if nT == 0:
            continue
        t = target[b, :nT]
        t[:, 1:] *= nG
        nGT += nT

        # Convert to position relative to box
        gx, gy, gw, gh = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        gi = torch.clamp(gx.long(), min=0, max=nG - 1)
        gj = torch.clamp(gy.long(), min=0, max=nG - 1)
        # Calculate ious between ground truth and each of the 3 anchors

        tb = torch.zeros(nT, 4)  # target boxes
        tb[:, 0] = gx - gw / 2
        tb[:, 1] = gy - gh / 2
        tb[:, 2] = gx + gw / 2
        tb[:, 3] = gy + gh / 2

        # iou of targets-anchors (using wh only)
        box1 = t[:, 3:5]
        box2 = anchor_xywh[:, gj, gi, 2:]
        inter_area = torch.min(box1, box2).prod(2)
        iou_anch = inter_area / (gw*gh + box2.prod(2) - inter_area + 1e-16)

        # iou of targets-predictions
        iou_pred = torch.zeros(nT, nA)  # iou of predicted boxes
        #for i in range(nA):
            #iou_pred[:, i] = bbox_iou(tb, pred_boxes[b, i, gj, gi])

        # Select best iou_pred and anchor
        iou_pred, _ = iou_pred.max(1)
        iou_anch_best, a = iou_anch.max(0)  # best anchor [0-2] for each target

        # Two targets can not claim the same anchor
        if nT > 1:
            u = np.concatenate((gi.numpy(), gj.numpy(), a.numpy()), 0).reshape(3, -1)
            iou_anch_order = np.argsort(-iou_anch_best)  # best to worst
            _, first_unique = np.unique(u[:, iou_anch_order], axis=1, return_index=True)  # first unique indices
            i = iou_anch_order[first_unique]
            a, gj, gi, t, iou_pred = a[i], gj[i], gi[i], t[i], iou_pred[i]
        else:
            i = 0

        tc, gx, gy, gw, gh = t[:, 0].long(), t[:, 1], t[:, 2], t[:, 3], t[:, 4]

        # Coordinates
        tx[b, a, gj, gi] = gx - gi.float()
        ty[b, a, gj, gi] = gy - gj.float()
        # print(len(torch.nonzero(tx - A)), nT)

        # Width and height
        tw[b, a, gj, gi] = gw / anchor_wh[a, 0] / 3
        th[b, a, gj, gi] = gh / anchor_wh[a, 1] / 3
        # tw[b, a, gj, gi] = torch.log(gw / anchor_wh[a, 0] + 1e-16)
        # th[b, a, gj, gi] = torch.log(gh / anchor_wh[a, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1
        # predicted classes and confidence
        # pcls = torch.argmax(pred_cls[b, a, gj, gi], 1)
        # pconf = pred_conf[b, a, gj, gi]

        #TP[b, i] = ((iou_pred > 0.5) & (pconf > 0.5) & (pcls == tc)).float()
        #FP[b, i] = ((iou_pred > 0.5) & (pconf > 0.5) & (pcls != tc)).float()
        #FN[b, :nT] = 1.0
        #FN[b, i] = ((TP[b, i] == 0) & (FP[b, i] == 0)).float()

    # precision = TP.sum() / (TP + FP + 1e-16).float()
    # recall = TP.float() / (TP + FN + 1e-16).float()
    # ap = nTP / nGT  # compute_ap(recall, precision)
    ap = 0
    return nGT, ap, tx, ty, tw, th, tconf == 1, tcls, TP, FP, FN


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='uint8')[y])


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

        # suppress boxes from other classes
        thresh = 0.25

        a = output[image_i]
        a = a[np.argsort(-a[:, 5])]  # sort best to worst

        radius = 20  # area to search for cross-class ious
        for i in range(len(a)):
            if i >= len(a) - 1:
                break
            close = torch.nonzero(
                (abs(a[i, 0] - a[i + 1:, 0]) < radius) & (abs(a[i, 1] - a[i + 1:, 1]) < radius)) + i + 1
            if len(close) > 0:
                iou = bbox_iou(a[i:i + 1, :4], a[close.squeeze(), :4].reshape(-1, 4))
                bad = close[iou > thresh]
                if len(bad) > 0:
                    mask = torch.ones(len(a)).type(torch.ByteTensor)
                    mask[bad] = 0
                    a = a[mask]

        # if prediction.is_cuda:
        #    a = a.cuda()
        output[image_i] = a

    return output

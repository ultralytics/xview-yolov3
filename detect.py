from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import random
import datetime
import argparse
import cv2

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('-config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('-weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('-class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('-img_size', type=int, default=416, help='size of each image dimension')
opt = parser.parse_args()
print(opt)

# @profile
def main(opt):
    os.makedirs('output', exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)
    model.to(device)
    model.eval()

    dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    classes = load_classes(opt.class_path)  # Extracts class labels from file


    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print('\nPerforming object detection:')
    prev_time = time.time()
    for batch_i, (img_paths, im) in enumerate(dataloader):
        # Configure input
        im = im.type(Tensor)

        # Get detections
        with torch.no_grad():
            detections = model(im)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        imgs.extend(img_paths)
        img_detections.extend(detections)


    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print('\nSaving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))

        # read image
        img = cv2.imread(path)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                x1y1x2y2 = torch.Tensor([x1, y1, x1 + box_w, y1 + box_h])

                # Add the bbox to the plot
                plot_one_box(x1y1x2y2, img, label=classes[int(cls_pred)])

            # Save generated image with detections
            cv2.imwrite('output/' + path.split('/')[-1], img)


if __name__ == '__main__':
    main(opt)

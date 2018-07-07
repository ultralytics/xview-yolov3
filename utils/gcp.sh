#!/usr/bin/env bash

# reattach screen
# screen -r

# detach from screen
# ctrl + a + d

# clone repo
sudo rm -rf yolo
git clone https://github.com/ultralytics/yolo

# do training
mkdir yolo/checkpoints
cp fresh2wC_best_608.pt yolo/checkpoints
cp printedResults.txt yolo
cd yolo
python3 train.py -epochs 9999

# move checkpoint to fuse drive
# google-drive-ocamlfuse drive
# mv yolov3/checkpoints/epoch0.pt drive


# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1530739498&Signature=ZbsIv1dxUTUceGBI-lGv7EeA~oHrBJnlLxwsds38FA1MBKdkrfMJnpntRYFLDqS04~27ps55NzGczHsLMd1xCMCvx4JYnqM0~xuONbZaqYzabTnqWc0RdJ4SNVxWVDgq8YFF4kq5yfpGA0oxFd8JtAZD9FG12eq-uGqFXYHc1JjVAT8OE9-usj5CkhMbobUsQBfeCCKaLitJoCAuAIuxbsgGr6YWQ~mPyyIk2-uWXOJSXJyKhlrjSKwXRhZL2TYZ2N~WG5IB1DpusIh4gg0IqntkszrcmT8VCmc7RfsNZMxBSFerHWpETsijq4LrWwRfA4kSwvr9D4W7TNljPcZ8wA__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz
sudo rm -rf train_images/._*
# lastly convert each .tif to a .bmp for faster loading in cv2

from utils import datasets
datasets.convert_tif2bmp('../train_images')
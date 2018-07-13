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
#cp fresh3.pt yolo/checkpoints
#cp results.txt yolo
cd yolo
python3 train.py -epochs 9999

python3 detect.py -config_path 'cfg/yolovx_YL0.cfg' -weights_path 'checkpoints/fresh9.pt' -conf_thres 0.999

# move checkpoint to fuse drive
# google-drive-ocamlfuse drive
# mv yolov3/checkpoints/epoch0.pt drive


# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1531465860&Signature=I0E6LJd3cfb0N7K~nbZWsO6twMHEDMCOVP1t5VCFAW2kicktnfaHhtgh-AXtOPaPSQzR57PpEYW3GnjunDAMJxq-7DhRyGTYGrPlgaVlccWIGFGzH6itlWLvmLDJCuVEgxtscBkHSvZPdfr46~QH-KNbXt-Cz4N~X7FY2EAJ-xzKAzTz3gIhv-bpI~tc6uHAxcdcyu6wiMv8F3sB0LfYrlpytAUvEA3-h9XEigPlcHAX1eMGRfBsDAFteSeziw8ZPOFYy~~agRW9mZe7PP60y8SCBH9dImehylAmQOZ2AYTamWZYI-8~20Lzaa8GmjBKy9Y9YL6LnBVNCPm2wqf4Tw__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz
sudo rm -rf train_images/._*
sudo shutdown
# lastly convert each .tif to a .bmp for faster loading in cv2

cd yolo
python3
from utils import datasets
datasets.convert_tif2bmp('../train_images')
sudo shutdown now

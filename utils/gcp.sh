#!/usr/bin/env bash

# reattach screen
# screen -r

# detach from screen
# ctrl + a + d

# clone repo
sudo rm -rf drive
mkdir drive
google-drive-ocamlfuse drive

sudo rm -rf yolo
git clone https://github.com/ultralytics/yolo
# do training
mkdir yolo/checkpoints
cp drive/fresh9_e450.pt yolo/checkpoints
#cp results.txt yolo
cd yolo
python3 train.py -epochs 9999

python3 detect.py -config_path 'cfg/yolovx_YL0.cfg' -weights_path 'checkpoints/fresh9.pt' -conf_thres 0.999

# move checkpoint to fuse drive
# google-drive-ocamlfuse drive
# mv yolov3/checkpoints/epoch0.pt drive


# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1531850600&Signature=FvYp1qDdadcfOAF2ELmHSJXaRYkq~R2KrRi0Fk3akL1~UZWiCm26QjLh52e11Ga-99GNAkLylXauCgt0k17hmw2aMrMDW-z9Qo9hGQV-BkYEimhd~dyybOqqGJ3ZWG3CmeesHHJ7ScdDpv9aIxZTNo-QUSABA8g5X2oMs96RWOy-GnAw09W8liBIoLAfeoGcqOubvY7vOMtFeFgFatzmMSoLPQ-Y8Zv2bGpQyih-pd7A2S0VAE3ccDwvjKgdOgYeuZLXBNaF5Wy~-JNX2RdaqaXmLO42P3soxT5FnCnGbLYoVAI7K6-mtlttcw0VOTMXqWvoN8QOsdZenREhKfJ0iw__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz
sudo rm -rf train_images/._*

# convert all .tif to .bmp
sudo rm -rf yolo
git clone https://github.com/ultralytics/yolo
cd yolo
python3
from utils import datasets
datasets.convert_tif2bmp('../train_images')
sudo shutdown now


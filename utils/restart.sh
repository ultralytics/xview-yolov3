#!/usr/bin/env bash

# reattach screen
# screen -r

# detach from screen
# ctrl + a + d

cp yolo/checkpoints/c60.pt . && cp yolo/results.txt .

# clone repo
sudo rm -rf drive
mkdir drive
google-drive-ocamlfuse drive

sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo
# do training
# mkdir yolo/checkpoints
# cp c60.pt yolo/checkpoints/restart.pt
cp results.txt yolo/results.txt
cd yolo
python3 train.py -epochs 9999 -img_size 608


# python3 detect.py -config_path 'cfg/c60.cfg' -weights_path 'checkpoints/fresh9.pt' -conf_thres 0.999

# move checkpoint to fuse drive
# google-drive-ocamlfuse drive
# mv yolov3/checkpoints/epoch0.pt drive


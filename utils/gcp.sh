#!/usr/bin/env bash

## move training data out
#sudo mv yolov3/train_images ..
#sudo mv yolov3/train_labels ..
#
##clone repo
cd ..
sudo rm -rf yolov3
git clone https://github.com/ultralytics/yolov3

# do training
cd yolov3
python3  train.py -img_size 416 -batch_size 4 -epochs 10

# move training data out
#sudo mv train_images ..
#sudo mv train_labels ..

#sudo shutdown
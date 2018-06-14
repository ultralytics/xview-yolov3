#!/usr/bin/env bash

## move training data out
#sudo mv yolov3/train_images ..
#sudo mv yolov3/train_labels ..
#
##clone repo
#sudo rm -rf yolov3
git clone https://github.com/ultralytics/yolov3
#
## move training data to folder
sudo mv train_images yolov3
sudo mv train_labels yolov3

# do training
python3  train_xview.py -img_size 864 -batch_size 4 -epochs 10

# move training data out
#sudo mv train_images ..
#sudo mv train_labels ..

#sudo shutdown
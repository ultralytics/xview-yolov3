#!/usr/bin/env bash

# reattach screen
# screen -r

# detach from screen
# ctrl + a + d

# clone repo
#cd ..
sudo rm -rf yolo
git clone https://github.com/ultralytics/yolo

# do training
cd yolo
python3 train.py -batch_size 2 -img_size 1056

# move checkpoint to fuse drive
#mv yolov3/checkpoints/epoch0.pt drive


#sudo shutdown
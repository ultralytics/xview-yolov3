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
python3  train.py -img_size 416 -batch_size 4 -n_cpu 2 -epochs 250 -checkpoint_interval 50

# move checkpoint to fuse drive
#mv yolov3/checkpoints/epoch0.pt drive


#sudo shutdown
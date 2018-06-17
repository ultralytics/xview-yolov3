#!/usr/bin/env bash

# reattach screen
# screen -r

# detach from screen
# ctrl + a + d

# clone repo
#cd ..
sudo rm -rf yolov3
git clone https://github.com/ultralytics/yolov3

# do training
cd yolov3
python3  traing.py -img_size 864 -batch_size 4 -n_cpu 8 -epochs 360 -checkpoint_interval 40

# move checkpoint to fuse drive
#mv yolov3/checkpoints/epoch0.pt drive


#sudo shutdown
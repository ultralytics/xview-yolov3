#!/usr/bin/env bash

# restart
cp yolo/checkpoints/*.pt restart.pt && cd yolo && python3 train.py -epochs 300 -img_size 1024

# clone repo
sudo rm -rf drive
mkdir drive
google-drive-ocamlfuse drive

cp fresh9_5_e201.pt latest.pt
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo && python3 train.py -epochs 999 -img_size 608 -cfg cfg/c60_a9.cfg


sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo
# do training
# mkdir yolo/checkpoints
# cp c60.pt yolo/checkpoints/restart.pt
# wget https://storage.googleapis.com/ultralytics/fresh9_5_e201.pt
# cp results.txt yolo/results.txt
cd yolo
python3 train.py -epochs 999 -img_size 608


python3 detect.py -config_path 'cfg/c60.cfg' -weights_path 'checkpoints/fresh9.pt' -conf_thres 0.999

# move checkpoint to fuse drive
# google-drive-ocamlfuse drive
# mv yolov3/checkpoints/epoch0.pt drive


# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1532553970&Signature=C-W3dFvU-ygEp2lsWRM~RJTFEKahLhFf1veTnsSUuhR0KPZUZP40ooXOJfiBuUg4rNG2rkIw~fthM0YURlivWcaz6dexRsA2VowIqISNyIEWm~0qu983Wog2LE41ZzXWGk8el2fkwDBa~bh9DAOYYhk7OKkfS7Xfzj3a1w1bZ1x7kkSBzc3YjnaKIdqBuAg-1lk~OVzBaGp8B3wBJbYGHf77~IESSu6Zd4-AcGDATjr~XpByqj1LxeDyl84-3~bUvsGqlBqnquvJVndvYYAfn4gFzDu0CNm3hsu9YQk5oCimCcbySDXQ3rnldJPLgTw8pqKzJUEGDuq7tx6fTLblvw__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz
sudo rm -rf train_images/._* train_images/659.tif train_images/769.tif

# convert all .tif to .bmp
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo

python3
from utils import datasets
datasets.convert_tif2bmp_clahe('../train_images')
exit()


cd train_images
rm -rf 149.bmp
rm -rf 374.bmp
rm -rf 401.bmp
rm -rf 422.bmp
rm -rf 437.bmp
rm -rf 457.bmp
rm -rf 517.bmp
rm -rf 541.bmp
rm -rf 567.bmp
rm -rf 595.bmp
rm -rf 601.bmp
rm -rf 605.bmp
rm -rf 606.bmp
rm -rf 611.bmp
rm -rf 626.bmp
rm -rf 650.bmp
rm -rf 671.bmp
rm -rf 672.bmp
rm -rf 674.bmp
rm -rf 680.bmp
rm -rf 791.bmp
rm -rf 813.bmp
rm -rf 817.bmp
rm -rf 930.bmp
rm -rf 1155.bmp
rm -rf 1311.bmp
rm -rf 1353.bmp
rm -rf 1440.bmp
rm -rf 1453.bmp
rm -rf 1647.bmp
rm -rf 1888.bmp
rm -rf 2247.bmp
rm -rf 2495.bmp
rm -rf 2503.bmp
cd ..

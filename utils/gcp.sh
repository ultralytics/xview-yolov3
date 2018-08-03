#!/usr/bin/env bash

# Start
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo && python3 train.py -epochs 999 -img_size 608
sudo rm -rf mnist && git clone https://github.com/ultralytics/mnist && cd mnist && python3 train_xview_classes.py


# Resume
cd yolo && python3 train.py -epochs 999 -img_size 608 -resume 1

# Detect
gsutil cp gs://ultralytics/fresh9_5_e201.pt yolo/checkpoints
gsutil cp gs://ultralytics/6layer_submit.pt yolo/checkpoints/classifier.pt
cd yolo && python3 detect2.py

# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1533331306&Signature=BPSqQT6vJLeTr30vbQtjuAhbCEASZM0X8bIfyxZkvnzslN~n6QFpPEpgXnAuHCL72ZxFf6LxlymzH7L58KG1SnTtWejodh5Swm2IAeM3MTvP2vbOnFuoQMiIK1WJ4jhdSWZkekl0Gxbhq0p0vLJDW~qzCILG0OxEx7FZWwUe7za0o9JFrEZwwUIwNrpp~iPAxPG-UAowOPa-TLqLMPPqgQCUlTNhp~cE6uBBiKhovvSGZzIwNtct-hkm3AdsDpj1Vk2TOK3g7QzU4mdLn0I~9LOEygQR5i96DZFW7xg367n8rWxM7T6lM~YsOehKGhQ~xIohY5upawv6MCu62b1bWQ__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz
sudo rm -rf train_images/._* #train_images/659.tif train_images/769.tif

# convert all .tif to .bmp
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo

cd yolo
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

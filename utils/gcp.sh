#!/usr/bin/env bash

# Start
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo && python3  # from utils import utils; utils.createChips()
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo && python3 train.py -epochs 999 -img_size 608
sudo rm -rf mnist && git clone https://github.com/ultralytics/mnist && cd mnist && python3 train_xview_classes.py -name 'chips_20pad_square'


# Resume
cd yolo && python3 train.py -epochs 999 -img_size 608 -resume 1

# Detect
gsutil cp gs://ultralytics/fresh9_5_e201.pt yolo/checkpoints
gsutil cp gs://ultralytics/6layer_submit.pt yolo/checkpoints/classifier.pt
cd yolo && python3 detect.py

# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1533363111&Signature=oDeseDqRFNm0QS9RXrmy7VSnLzdpTQfx4Q~GFxyx~KCVay8l7JGcToOBp~GYMayA2-fy7pmjUphKJHghxfGt1Pyf566WkOq8b-OzbPbV99dEljQ23Gkwn7ndd0nULW3-mz2FDyPnrEM-LDlfUEC-npknbJc8S~~1I5LeJ48q51ZSlHOJn4bQUFfFzllPHd1YBomYm645KMS-yG185werTV0taJUqLvdRkqDWFzWfuNvEpXtCnUOdsS8DSAT9SqP81qUz81qAjHC2Wq-fRj2gYhpnqBixm3Y4Ng~O58QXCrhujkVgaXCrkFgc3cmAHdtC8qmegN50PBiSJAIsZxIwJg__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
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

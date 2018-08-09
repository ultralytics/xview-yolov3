#!/usr/bin/env bash

# Start
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo && python3  # from utils import utils; utils.createChips()
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo && python3 train.py -epochs 999
sudo rm -rf mnist && git clone https://github.com/ultralytics/mnist && cd mnist && python3 train_xview_classes.py -name 'chips_20pad_square'

# Resume
cd yolo && python3 train.py -epochs 999 -img_size 608 -resume 1

# Detect
gsutil cp gs://ultralytics/fresh9_5_e201.pt yolo/checkpoints
gsutil cp gs://ultralytics/6leaky681_stripped.pt yolo/checkpoints/classifier.pt
cd yolo && python3 detect.py -secondary_classifier 1

# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1533801227&Signature=T~bp20LvjNmcwtFJ-l4Cwuz-PN4Gbcf~61wGnDrW7ng8ApazzkQ66DzXdvod5IXpQWNUD~dC28iSKP1rDi~cLCgq4v~7JZ-5Tb1j1fTHC7xka27OYZKcubu6gt8TOkxArmy~n6xqy2PSj~n-m6NIhxcUbXRWyWbJqkj5X3w70Qfz3XlK0lxa5nFfeeiRLuUvmAP7Z6QObGLderlG72YbWM2eNRLjmsqUwMcLJMCuAZ5TKii9-PY8nld8F6u0cleq1rQxQ-r5ngTfPESomw98Qmj~87kTECh4Aw9Kw4N4oWGZfc~~iM50tgB4DONNmDfHrS4xlJXzAHA7aoMgQJibfA__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz && sudo rm -rf train_images/._*

# convert all .tif to .bmp
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo

python3
from utils import datasets
datasets.convert_tif2bmp('../train_images')
exit()


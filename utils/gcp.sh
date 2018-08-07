#!/usr/bin/env bash

# Start
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo && python3  # from utils import utils; utils.createChips()
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo && python3 train.py -epochs 999 -resume 1
sudo rm -rf mnist && git clone https://github.com/ultralytics/mnist && cd mnist && python3 train_xview_classes.py -name 'chips_20pad_square'

# Resume
cd yolo && python3 train.py -epochs 999 -img_size 608 -resume 1

# Detect
gsutil cp gs://ultralytics/fresh9_5_e201.pt yolo/checkpoints
gsutil cp gs://ultralytics/6leaky681_stripped.pt yolo/checkpoints/classifier.pt
cd yolo && python3 detect.py -secondary_classifier 1

# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1533700896&Signature=Kp4Ndquc1o7j4wBbiGOMVQDC8ihj2zIx3TCQF39DbMWY-Z~d3HyVmhZY99~JaZ1kxu7xRgegJhL-Uu-DkLyFaoJpqQR8EMnp-d2jc2fLwA6gsfNmwXSSKJXsMNEprw3ETe75Izi897xww4vrDzRvf-towkM6ExLzRa3X6F2sR6RIfcwo4yzowmjbOQlKHMMdSDmzsdXWOZSYmaa2FvT1ILFzjJYB5-X9i6hA5QH5S5ZkDiECC-ALj7vg-GI8W7TVDri2JfrizHUlqs9953AWigtKBK3ku-uRRXzhlrYqfGrTrhW17rRvz39GzUBXISKxCTx595GgReGLlSOC9ruLHA__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz && sudo rm -rf train_images/._*

# convert all .tif to .bmp
sudo rm -rf yolo && git clone https://github.com/ultralytics/yolo && cd yolo

python3
from utils import datasets
datasets.convert_tif2bmp('../train_images')
exit()

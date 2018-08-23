
<img src="https://storage.googleapis.com/ultralytics/UltralyticsLogoName1000×676.png" width="200">  

# Introduction

This directory contains software developed by Ultralytics LLC. For more information on Ultralytics projects please visit:
http://www.ultralytics.com  

# Description

The https://github.com/ultralytics/xview-yolo repo contains code to train YOLOv3 on the xView training set for the xView challenge: https://challenge.xviewdataset.org/. Credit to P.J. Reddie for YOLO (https://pjreddie.com/darknet/yolo/) and to Erik Lindernoren for the pytorch implementation (https://github.com/eriklindernoren/PyTorch-YOLOv3).

# Requirements

Python 3.6 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `scipy`
- `torch`
- `opencv-python`
- `h5py`
- `tqdm`

# Running

- Run `train.py` to begin training. Checkpoints will be saved in `/checkpoints` directory. Run `detect.py` to apply trained weights to an xview image, such as `5.tif` from the training set, shown below.

![Alt](https://github.com/ultralytics/xview/blob/master/output/5.jpg "example")

# Contact

For questions or comments please contact Glenn Jocher at glenn.jocher@ultralytics.com or visit us at http://www.ultralytics.com/contact
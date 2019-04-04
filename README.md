<img src="https://storage.googleapis.com/ultralytics/logo/logoname1000.png" width="200">

# Introduction

This directory contains software developed by Ultralytics LLC, and **is freely available for redistribution under the MIT license**. For more information on Ultralytics projects please visit:
http://www.ultralytics.com.

# Description

The https://github.com/ultralytics/xview-yolov3 repo contains code to train YOLOv3 on the xView training set for the xView challenge: https://challenge.xviewdataset.org/. **Credit to Joseph Redmon for YOLO** (https://pjreddie.com/darknet/yolo/) and to **Erik Lindernoren for the pytorch implementation** this work is based on (https://github.com/eriklindernoren/PyTorch-YOLOv3).

# Requirements

Python 3.6 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `scipy`
- `torch`
- `opencv-python`
- `h5py`
- `tqdm`

# Download Data

Download xView data from https://challenge.xviewdataset.org/data-download.

# Training

Before training, targets are cleaned up, removing outliers via sigma-rejection and creating 30 new k-means anchors for `c60_a30symmetric.cfg` with the MATLAB file `utils/analysis.m`:

<img src="https://github.com/ultralytics/xview-yolov3/blob/master/cfg/c60_a30.png" width="400">

**Start Training:** Run `train.py` to begin training after downloading xView data with and specifying xView path on line 41 (local) or line 43 (cloud).

**Resume Training:** Run `train.py -resume 1` to resume training from the most recently saved checkpoint `latest.pt`.

Each epoch consists of processing 8 608x608 sized chips randomly sampled from each (augmented) image at full resolution. An Nvidia GTX 1080 Ti will run about 100 epochs per day. Loss plots for the bounding boxes, objectness and class confidence should appear similar to results shown here. **Note that overtraining starts to become a significant issue past about 200 epochs, a problem I was not able to overcome during the competition.** Best validation mAP is 0.16 after 300 epochs (3 days), corresponding to a training mAP of 0.30.

![Alt](https://github.com/ultralytics/xview-yolov3/blob/master/data/xview_training_loss.png "training loss")

## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the full-resolution input images in accordance with the following specifications. 8 608 x 608 sized chips are then selected at random from the augmented image for training. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images.

Augmentation | Description
--- | ---
Translation | +/- 1% (vertical and horizontal)
Rotation | +/- 20 degrees
Shear | +/- 3 degrees (vertical and horizontal)
Scale | +/- 30%
Reflection | 50% probability (vertical and horizontal)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%

# Inference

Checkpoints will be saved in `/checkpoints` directory. Run `detect.py` to apply trained weights to an xView image, such as `5.tif` from the training set, shown here.

![Alt](https://github.com/ultralytics/xview-yolov3/blob/master/output/5.jpg "example")

# Citation

[![DOI](https://zenodo.org/badge/137117503.svg)](https://zenodo.org/badge/latestdoi/137117503)

# Contact

Issues should be raised directly in the repository. For additional questions or comments please email Glenn Jocher at glenn.jocher@ultralytics.com or visit us at https://contact.ultralytics.com/contact.


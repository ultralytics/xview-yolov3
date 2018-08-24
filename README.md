<img src="https://storage.googleapis.com/ultralytics/UltralyticsLogoName1000×676.png" width="200">  

# Introduction

This directory contains software developed by Ultralytics LLC. For more information on Ultralytics projects please visit:
http://www.ultralytics.com  

# Description

The https://github.com/ultralytics/xview-yolov3 repo contains code to train YOLOv3 on the xView training set for the xView challenge: https://challenge.xviewdataset.org/. Credit to P.J. Reddie for YOLO (https://pjreddie.com/darknet/yolo/) and to Erik Lindernoren for the pytorch implementation (https://github.com/eriklindernoren/PyTorch-YOLOv3).

# Requirements

Python 3.6 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `scipy`
- `torch`
- `opencv-python`
- `h5py`
- `tqdm`

# Running

Before training, targets are cleaned up, removing outlies via sigma-rejection and creating 30 new k-means anchors for `c60_a30symmetric.cfg` with the MATLAB file `analysis.m`:
![Alt](https://github.com/ultralytics/xview-yolov3/blob/master/cfg/c60_a30.png "30 kmeans xView anchors")

Run `train.py` to begin training. Note that `train.py` will look for a folder with xView training images at the path specified on line 41. Each epoch consists of processing 8 608x608 sized chips randomly sampled from each (augmented) image at full resolution. An Nvidia GTX 1080 Ti will run about 100 epochs per day. Loss plots for the bounding boxes, objectness and class confidence should appear similar to results shown here. **Note that overtraining starts to become a significant issue past about 200 epochs, an issue I was not able to overcome during the competition.** The best validation mAP this produces is 0.16 after about 300 epochs (3 days), corresponding to a training mAP of about 0.30.
![Alt](https://github.com/ultralytics/xview-yolov3/blob/master/data/xview_training_loss.png "training loss")

Checkpoints will be saved in `/checkpoints` directory. Run `detect.py` to apply trained weights to an xView image, such as `5.tif` from the training set, shown here.
![Alt](https://github.com/ultralytics/xview-yolov3/blob/master/output/5.jpg "example")

# Contact

For questions or comments please contact Glenn Jocher at glenn.jocher@ultralytics.com or visit us at http://www.ultralytics.com/contact
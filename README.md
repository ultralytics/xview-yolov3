<img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="400">

# Introduction

The https://github.com/ultralytics/xview-yolov3 repo contains code to train YOLOv3 on the xView training set for the xView challenge: https://challenge.xviewdataset.org/. **Credit to Joseph Redmon for YOLO**.

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/26833433/238799379-bb3b02f0-dee4-4e67-80ae-4b2378b813ad.jpg?raw=true" width="900">

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

<img src="https://github.com/ultralytics/xview-yolov3/blob/master/cfg/c60_a30.png?raw=true" width="400">

**Start Training:** Run `train.py` to begin training after downloading xView data with and specifying xView path on line 41 (local) or line 43 (cloud).

**Resume Training:** Run `train.py -resume 1` to resume training from the most recently saved checkpoint `latest.pt`.

Each epoch consists of processing 8 608x608 sized chips randomly sampled from each (augmented) image at full resolution. An Nvidia GTX 1080 Ti will run about 100 epochs per day. Loss plots for the bounding boxes, objectness and class confidence should appear similar to results shown here. **Note that overtraining starts to become a significant issue past about 200 epochs.** Best validation mAP is 0.16 after 300 epochs (3 days), corresponding to a training mAP of 0.30.

<img src="https://github.com/ultralytics/xview-yolov3/blob/master/data/xview_training_loss.png?raw=true" width="900">

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

<img src="https://github.com/ultralytics/xview/blob/master/output_img/1047.jpg?raw=true" width="900">

# Citation

[![DOI](https://zenodo.org/badge/137117503.svg)](https://zenodo.org/badge/latestdoi/137117503)

# Contact

For Ultralytics bug reports and feature requests please visit [GitHub Issues](https://github.com/ultralytics/ultralytics/issues), and join our [Discord](https://ultralytics.com/discord) community for questions and discussions!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

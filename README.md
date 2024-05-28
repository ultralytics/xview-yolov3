<br>
<img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320">

# 🚀 Introduction

Welcome to the [Ultralytics xView YOLOv3](https://github.com/ultralytics/xview-yolov3) repository! Here we provide code to train the powerful YOLOv3 object detection model on the xView dataset for the [xView Challenge](https://challenge.xviewdataset.org/). This challenge focuses on detecting objects from satellite imagery, advancing the state of the art in computer vision applications for remote sensing.

[![Ultralytics Actions](https://github.com/ultralytics/xview-yolov3/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/xview-yolov3/actions/workflows/format.yml) <a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/26833433/238799379-bb3b02f0-dee4-4e67-80ae-4b2378b813ad.jpg?raw=true" width="100%">

# 📦 Requirements

To run this project, ensure that you have Python 3.6 or later. You will also need to install several dependencies which can be done easily using pip:

```bash
pip3 install -U -r requirements.txt
```

The following packages should be included:

- `numpy`: For numerical operations.
- `scipy`: Useful for scientific and technical computations.
- `torch`: The PyTorch machine learning framework.
- `opencv-python`: Open Source Computer Vision Library.
- `h5py`: For managing and manipulating data in HDF5 format.
- `tqdm`: For adding progress bars to loops and command line.

# 📥 Download Data

Start by downloading the xView data from the [data download page](https://challenge.xviewdataset.org/data-download) of the xView Challenge.

# 🏋️‍♂️ Training

## Preprocessing Steps

Before we launch into training, we perform preprocessing on the targets to clean them up:

1. Outliers are removed using sigma-rejection.
2. A new set of 30 k-means anchors are created specifically for `c60_a30symmetric.cfg` using the MATLAB script `utils/analysis.m`:

<img src="https://github.com/ultralytics/xview-yolov3/blob/main/cfg/c60_a30.png?raw=true" width="500">

## Starting the Training

**To start training**, execute `train.py` after you have downloaded the xView data. You'll need to specify the path to your xView data on line 41 (for local execution) or line 43 (if you're working in the cloud).

## Resuming Training

**To resume training**, use the following command:

```bash
train.py --resume 1
```

Training will continue from the most recent checkpoint found in the `latest.pt` file.

During training, each epoch will process 8 randomly sampled 608x608 chips from each full-resolution image. If you're using a GPU like the Nvidia GTX 1080 Ti, you can expect to complete around 100 epochs per day.

Watch out for overtraining! It becomes a significant problem after roughly 200 epochs. The best validation mean Average Precision (mAP) observed is 0.16 after 300 epochs, which takes about 3 days, corresponding to a training mAP of 0.30.

You'll see loss plots for bounding boxes, objectness, and class confidence that should resemble the following results:

<img src="https://github.com/ultralytics/xview-yolov3/blob/main/data/xview_training_loss.png?raw=true" width="100%">

### Image Augmentation 📸

During training, `datasets.py` will apply various augmentations to the full-resolution input images using OpenCV. Here are the specifications for each augmentation applied:

| Augmentation   | Description                               |
| -------------- | ----------------------------------------- |
| Translation    | +/- 1% (vertical and horizontal)          |
| Rotation       | +/- 20 degrees                            |
| Shear          | +/- 3 degrees (vertical and horizontal)   |
| Scale          | +/- 30%                                   |
| Reflection     | 50% probability (vertical and horizontal) |
| HSV Saturation | +/- 50%                                   |
| HSV Intensity  | +/- 50%                                   |

Please note that augmentation is applied **only** during training and not during inference. All corresponding bounding boxes are automatically adjusted to match the augmented images.

# 🔍 Inference

Once training is done, model checkpoints will be available in the `/checkpoints` directory. Use `detect.py` to apply your trained weights to any xView image—for instance, `5.tif` from the training set:

<img src="https://github.com/ultralytics/xview/blob/main/output_img/1047.jpg?raw=true" width="100%">

# 📝 Citation

If you use this repository or the associated tools and datasets in your research, please cite accordingly:

[![DOI](https://zenodo.org/badge/137117503.svg)](https://zenodo.org/badge/latestdoi/137117503)

# 👥 Contribute

🤝 We love contributions from the community! Our open-source projects thrive on your help. To start contributing, please check out our [Contributing Guide](https://docs.ultralytics.com/help/contributing). Additionally, we'd love to hear from you through our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). It's a way to **impact** the future of our projects. A big shoutout and thank you 🙏 to all our contributors!

<!-- Image with SVG format can be troublesome in some markdown viewers -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" width="100%" alt="Ultralytics open-source contributors"></a>

# 📜 License

At Ultralytics, we provide two different licensing options to suit various use cases:

- **AGPL-3.0 License**: The [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) is an [OSI-approved](https://opensource.org/licenses/) open-source format that's best suited for students, researchers, and enthusiasts to promote collaboration and knowledge sharing. The full terms can be found in the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file.
- **Enterprise License**: If you're looking for a commercial application of our software and models, the Enterprise License enables integration into commercial products while bypassing the open-source stipulations of the AGPL-3.0. For embedding our solutions into your commercial offerings, please contact us through [Ultralytics Licensing](https://ultralytics.com/license).

# 📬 Contact

🐞 For reporting bugs or suggesting new features, please open an issue on our [GitHub Issues](https://github.com/ultralytics/xview-yolov3/issues) page. And if you have questions or fancy engaging with us, join our vibrant [Discord](https://ultralytics.com/discord) community!

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

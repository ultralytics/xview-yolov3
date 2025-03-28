<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 Introduction

Welcome to the Ultralytics `xview-yolov3` repository! This project provides the necessary code and instructions to train the powerful [Ultralytics YOLOv3](https://docs.ultralytics.com/models/yolov3/) object detection model on the challenging [xView dataset](https://challenge.xviewdataset.org/). The primary goal is to support participants in the [xView Challenge](https://challenge.xviewdataset.org/), which focuses on advancing the state-of-the-art in detecting objects within [satellite imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyse-satellite-imagery), a critical application of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) in remote sensing.

[![Ultralytics Actions](https://github.com/ultralytics/xview-yolov3/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/xview-yolov3/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/26833433/238799379-bb3b02f0-dee4-4e67-80ae-4b2378b813ad.jpg?raw=true" width="100%" alt="xView dataset example detections">

# 📦 Requirements

To successfully run this project, ensure your environment meets the following prerequisites:

- **Python:** Version 3.6 or later. You can download Python from the [official Python website](https://www.python.org/downloads/).
- **Dependencies:** Install the required packages using pip. It's recommended to use a virtual environment.

```bash
pip3 install -U -r requirements.txt
```

Key dependencies include:

- `numpy`: Essential for numerical operations in Python.
- `scipy`: Provides algorithms for scientific and technical computing.
- `torch`: The core [PyTorch](https://pytorch.org/) library for deep learning.
- `opencv-python`: The [OpenCV](https://opencv.org/) library for computer vision tasks.
- `h5py`: Enables interaction with data stored in HDF5 format.
- `tqdm`: A utility for displaying progress bars in loops and command-line interfaces.

# 📥 Download Data

Begin by downloading the necessary xView dataset files. You can obtain the data directly from the [xView Challenge data download page](https://challenge.xviewdataset.org/data-download). Ensure you have sufficient storage space, as satellite imagery datasets can be quite large.

# 🏋️‍♂️ Training

Training the YOLOv3 model on the xView dataset involves preprocessing the data and then running the training script.

## Preprocessing Steps

Before initiating the training process, we perform several preprocessing steps on the target labels to enhance model performance:

1.  **Outlier Removal:** Outliers in the dataset are identified and removed using sigma-rejection to clean the data.
2.  **Anchor Generation:** A new set of 30 [k-means anchors](https://www.ultralytics.com/glossary/anchor-based-detectors) are generated specifically tailored for the `c60_a30symmetric.cfg` configuration file. This process utilizes the MATLAB script `utils/analysis.m`. The generated anchors help the model better predict bounding boxes of various sizes and aspect ratios present in the xView dataset.

<img src="https://github.com/ultralytics/xview-yolov3/blob/main/cfg/c60_a30.png?raw=true" width="500" alt="k-means anchors plot">

## Starting the Training

Once the xView data is downloaded and placed in the expected directory, you can **start training** by executing the `train.py` script. You will need to configure the path to your xView data within the script:

- Modify line 41 for local machine execution.
- Modify line 43 if you are training in a cloud environment like [Google Colab](https://docs.ultralytics.com/integrations/google-colab/) or [Kaggle](https://docs.ultralytics.com/integrations/kaggle/).

```bash
python train.py
```

## Resuming Training

If your training session is interrupted, you can easily **resume training** from the last saved checkpoint. Use the `--resume` flag as shown below:

```bash
python train.py --resume 1
```

The script will automatically load the weights from the `latest.pt` file located in the `checkpoints/` directory and continue the training process.

## Training Details

During each training epoch, the system processes 8 randomly sampled 608x608 pixel chips extracted from each full-resolution image in the dataset. On hardware like an Nvidia GTX 1080 Ti, you can typically complete around 100 epochs per day.

Be mindful of [overfitting](https://www.ultralytics.com/glossary/overfitting), which can become a significant issue after approximately 200 epochs. Monitoring validation metrics is crucial. The best observed validation mean Average Precision ([mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)) in experiments was 0.16 after 300 epochs (roughly 3 days of training), corresponding to a training mAP of 0.30.

Monitor the training progress by observing the loss plots for bounding box regression, objectness, and class confidence. These plots should ideally show decreasing trends, similar to the example below:

<img src="https://github.com/ultralytics/xview-yolov3/blob/main/data/xview_training_loss.png?raw=true" width="100%" alt="xView training loss plot">

### Image Augmentation 📸

To improve model robustness and generalization, the `datasets.py` script applies various [data augmentations](https://www.ultralytics.com/glossary/data-augmentation) to the full-resolution input images during training using OpenCV. The specific augmentations and their parameters are:

| Augmentation   | Description                               |
| -------------- | ----------------------------------------- |
| Translation    | +/- 1% (vertical and horizontal)          |
| Rotation       | +/- 20 degrees                            |
| Shear          | +/- 3 degrees (vertical and horizontal)   |
| Scale          | +/- 30%                                   |
| Reflection     | 50% probability (vertical and horizontal) |
| HSV Saturation | +/- 50%                                   |
| HSV Intensity  | +/- 50%                                   |

**Note:** Augmentation is applied **only** during the training phase. During inference or validation, the original, unaugmented images are used. The corresponding [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates are automatically adjusted to match the transformations applied to the images. Explore more augmentation techniques with [Albumentations](https://docs.ultralytics.com/integrations/albumentations/).

# 🔍 Inference

After training completes, the model checkpoints (`.pt` files) containing the learned weights are saved in the `checkpoints/` directory. You can use the `detect.py` script to perform [inference](https://docs.ultralytics.com/modes/predict/) on new or existing xView images using your trained model.

For example, to run detection on the image `5.tif` from the training set using the best performing weights (`best.pt`), you would run:

```bash
python detect.py --weights checkpoints/best.pt --source path/to/5.tif
```

The script will process the image, detect objects, draw bounding boxes, and save the output image. An example output might look like this:

<img src="https://github.com/ultralytics/xview/blob/main/output_img/1047.jpg?raw=true" width="100%" alt="Example inference output on xView image">

# 📝 Citation

If you find this repository, the associated tools, or the xView dataset useful in your research or work, please consider citing the relevant sources:

[![DOI](https://zenodo.org/badge/137117503.svg)](https://zenodo.org/badge/latestdoi/137117503)

For the xView dataset itself, please refer to the citation guidelines provided on the [xView Challenge website](https://challenge.xviewdataset.org/).

# 👥 Contribute

🤝 We thrive on community contributions! Open-source projects like this benefit greatly from your input. Whether it's fixing bugs, adding features, or improving documentation, your help is valuable. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details on how to get started.

We also invite you to share your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). Your insights help shape the future of Ultralytics projects.

A huge thank you 🙏 to all our contributors for making our community vibrant and innovative!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

# 📜 License

Ultralytics offers two licensing options to accommodate different needs:

- **AGPL-3.0 License:** Ideal for students, researchers, and enthusiasts, this [OSI-approved](https://opensource.org/license/agpl-v3) open-source license promotes open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/xview-yolov3/blob/main/LICENSE) file for full details.
- **Enterprise License:** Designed for commercial use, this license allows integration of Ultralytics software and AI models into commercial products and services without the open-source requirements of AGPL-3.0. If your project requires an Enterprise License, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

# 📬 Contact

For bug reports, feature requests, or suggestions, please use the [GitHub Issues](https://github.com/ultralytics/xview-yolov3/issues) page. For general questions, discussions, and community interaction, join our [Discord](https://discord.com/invite/ultralytics) server!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageFolder(Dataset):  # for eval-only
    def __init__(self, path, img_size=416):
        try:
            if os.path.isdir(path):
                self.files = sorted(glob.glob('%s/*.*' % path))
            elif os.path.isfile(path):
                self.files = [path]
        except:
            print('Error: no files or folders found in supplied path.')

        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Add padding
        img = cv2.imread(img_path)
        img = resize_square(img, height=self.img_shape[0])[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

        # Normalize
        r, c = np.nonzero(img.sum(0))
        img -= img[:, r, c].mean()
        img /= img[:, r, c].std()
        return img_path, torch.from_numpy(img)

    def __len__(self):
        return len(self.files)


class ListDataset_xview(Dataset):  # for training
    def __init__(self, folder_path, img_size=416):
        p = folder_path + 'train_images3'
        self.img_files = sorted(glob.glob('%s/*.*' % p))
        assert len(self.img_files) > 0, 'No images found in path %s' % p
        self.img_shape = (img_size, img_size)
        self.label_files = [path.replace('_images3', '_labels').replace('.tif', '.txt') for path in self.img_files]
        self.max_objects = 5000
        self.mu = np.array([40.746, 49.697, 60.134])[:, np.newaxis, np.newaxis] / 255.0
        self.std = np.array([22.046, 24.498, 29.99])[:, np.newaxis, np.newaxis] / 255.0

    # @profile
    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)]
        img = cv2.imread(img_path)

        #small_path = img_path.replace('train_images','train_images_' + str(self.img_shape[0]))
        #if os.path.isfile(small_path):
        #     img = cv2.imread(img_path)
        #     shape = img.shape[:2]  # shape = [height, width]
        #     ratio = float(self.img_shape[0]) / max(shape)
        #     new_shape = [int(shape[0] * ratio), int(shape[1] * ratio)]
        #     dw = self.img_shape[0] - new_shape[1]  # width padding
        #     dh = self.img_shape[0] - new_shape[0]  # height padding
        #     top, bottom = dh // 2, dh - (dh // 2)
        #     left, right = dw // 2, dw - (dw // 2)
        #     img = cv2.resize(img, (new_shape[1], new_shape[0]),interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
        #     cv2.imwrite(small_path,img)
        # else:
        #    img = cv2.imread(small_path)

        # random_affine(img, points=None, degrees=(-5, 5), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        padded_h = max(h, w)
        padded_w = padded_h

        # Add padding
        img = resize_square(img, height=self.img_shape[0])[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

        # Normalize
        r, c = np.nonzero(img.sum(0))
        img -= img[:, r, c].mean()
        img /= img[:, r, c].std()

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)]

        labels = None
        if os.path.exists(label_path):
            # labels0 = np.loadtxt(label_path)  # slower than with open() as file:
            with open(label_path, 'r') as file:
                a = file.read().replace('\n', ' ').split()
            labels = np.array([float(x) for x in a]).reshape(-1, 5)

            # convert from x1y1x2y2 to xywh (required to convert xview to coco)
            cx = (labels[:, 1] + labels[:, 3]) / 2
            cy = (labels[:, 2] + labels[:, 4]) / 2
            cw = labels[:, 3] - labels[:, 1]
            ch = labels[:, 4] - labels[:, 2]
            labels[:, 1] = cx / w
            labels[:, 2] = cy / h
            labels[:, 3] = cw / w
            labels[:, 4] = ch / h
            # Extract coordinates for unpadded + unscaled image (coco)
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
            # remap xview classes 11-94 to 0-61
            labels[:, 0] = xview_classes2indices(labels[:, 0])
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5), dtype=np.float32)
        if labels is not None:
            nT = len(labels)  # number of targets
            filled_labels[range(nT)[:self.max_objects]] = labels[:self.max_objects]
        return img_path, torch.from_numpy(img), torch.from_numpy(filled_labels)

    def __len__(self):
        return len(self.img_files)


def xview_classes2indices(classes):  # remap xview classes 11-94 to 0-61
    indices = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11, 12, 13, 14,
               15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1, 29, 30, 31, 32, 33, 34,
               35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46, 47, 48, 49, 50, 51, 52, -1, 53, -1,
               -1, 54, 55, 56, -1, 57, -1, -1, 58, -1, 59, -1, 60, 61]
    return [indices[int(c)] for c in classes]


# @profile
def resize_square(img, height=416, color=(0, 0, 0)):  # resizes a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)
    new_shape = [int(shape[0] * ratio), int(shape[1] * ratio)]
    dw = height - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def random_affine(img, points=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    a = np.random.rand(1) * (degrees[1] - degrees[0]) + degrees[0]
    cx = img.shape[0] * (1 + np.random.rand(1) * translate[0])
    cy = img.shape[1] * (1 + np.random.rand(1) * translate[1])
    s = np.random.rand(1) * (scale[1] - scale[0]) + scale[0]

    M = cv2.getRotationMatrix2D(angle=a, center=(cy, cx), scale=s)
    imw = cv2.warpAffine(img, M, dsize=(img.shape[1], img.shape[0]))

    # Return warped points as well
    if points:
        M3 = np.concatenate((M, np.zeros((1, 3))), axis=0)
        M3[2, 2] = 1
        points_warped = 0
        # x = np.array([[1, .2], [.3, .5], [.7, .8]], dtype=np.float32) * 2000
        # M3 = M3.astype(np.float32)
        # points_warped = cv2.perspectiveTransform(np.array([x]), M)

        # import matplotlib.pyplot as plt
        # plt.imshow(imw)
        # plt.plot(x[0],x[1])
        return imw, points_warped
    else:
        return imw

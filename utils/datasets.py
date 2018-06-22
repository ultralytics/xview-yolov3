import glob
import os
import random

import cv2
import numpy as np
import scipy.io
import torch
from PIL import Image
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
        r, c = np.nonzero(img.sum(0))  # image must be [3, 416, 416] ordere here
        img[:, r, c] -= img[:, r, c].mean()
        img[:, r, c] /= img[:, r, c].std()
        return img_path, torch.from_numpy(img)

    def __len__(self):
        return len(self.files)


class ListDataset_xview():  # for training
    def __init__(self, folder_path, img_size=416):
        p = folder_path + 'train_images3'
        self.img_files = sorted(glob.glob('%s/*.*' % p))
        assert len(self.img_files) > 0, 'No images found in path %s' % p
        self.height = img_size
        # self.label_files = [path.replace('_images3', '_labels').replace('.tif', '.txt') for path in self.img_files]
        self.max_objects = 7607
        # load targets
        self.mat = scipy.io.loadmat('utils/targets.mat')
        self.mat['id'] = self.mat['id'].squeeze()
        # make folder for reduced size images
        self.small_folder = p + '_' + str(img_size) + '/'
        os.system('mkdir ' + self.small_folder)


        print('initialized!!')

    # @profile
    def __getitem__(self, index):
        index = index % len(self.img_files)
        img_path = self.img_files[index]

        # load labels
        chip = img_path.rsplit('/')[-1].replace('.tif', '')
        i = np.nonzero(self.mat['id'] == np.array(float(chip)))[0]
        labels = self.mat['targets'][i]

        # img = cv2.imread(img_path)
        # h, w, _ = img.shape
        small_path = self.small_folder + str(chip) + '.tif'
        if not os.path.isfile(small_path):
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            ratio = float(self.height) / max(h, w)
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)),
                             interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
            cv2.imwrite(small_path, img)
        else:
            img = cv2.imread(small_path)
            # load original image width and height
            if len(labels) > 0:
                w, h = self.mat['wh'][i[0]]
            else:
                w, h = Image.open(img_path).size

        # Add padding
        img = resize_square(img, height=self.height)
        ratio = float(self.height) / max(h, w)
        pad, padx, pady = (max(h, w) - min(h, w)) / 2, 0, 0
        if h > w:
            padx = pad
        elif h < w:
            pady = pad

        # label_path = self.label_files[index]
        # with open(label_path, 'r') as file:
        #    a = file.read().replace('\n', ' ').split()
        # labels = np.array([float(x) for x in a]).reshape(-1, 5)

        if len(labels) > 0:
            labels[:, [1, 3]] += padx
            labels[:, [2, 4]] += pady
            labels[:, 1:5] *= ratio

            # plot
            # import matplotlib.pyplot as plt
            # plt.imshow(img[0])
            # plt.plot(labels[:, 1], labels[:, 2], '.')
            # plt.plot(labels[:, 3], labels[:, 4], '.')

        # random affine
        # img, labels = random_affine(img, targets=labels, degrees=(-5, 5), translate=(.05, .05),  scale=(.95, 1.05))
        nL = len(labels)

        # convert labels to xywh
        if nL > 0:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / self.height

        # random lr flip
        # if random.random() > 0.5:
        #     img = np.fliplr(img)
        #     if nL > 0:
        #         labels[:, 1] = 1 - labels[:, 1]
        #
        # # random ud flip
        # if random.random() > 0.5:
        #     img = np.flipud(img)
        #     if nL > 0:
        #         labels[:, 2] = 1 - labels[:, 2]

        # random 90deg rotation
        # if random.random() > 0.5:
        #    img = np.rot90(img)

        # Normalize
        img = np.ascontiguousarray(img)
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        r, c = np.nonzero(img.sum(0))  # image must be [3, 416, 416] ordere here
        img[:, r, c] -= img[:, r, c].mean()
        img[:, r, c] /= img[:, r, c].std()

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5), dtype=np.float32)
        if nL > 0:
            # remap xview classes 11-94 to 0-61
            labels[:, 0] = xview_classes2indices(labels[:, 0])
            filled_labels[range(nL)[:self.max_objects]] = labels[:self.max_objects]

        return img_path, torch.from_numpy(img), torch.from_numpy(filled_labels)

    def __len__(self):
        return len(self.img_files)


def xyxy2xywh(box):
    xywh = np.zeros(box.shape)
    xywh[:, 0] = (box[:, 0] + box[:, 2]) / 2
    xywh[:, 1] = (box[:, 1] + box[:, 3]) / 2
    xywh[:, 2] = box[:, 2] - box[:, 0]
    xywh[:, 3] = box[:, 3] - box[:, 1]
    return xywh


def xview_classes2indices(classes):  # remap xview classes 11-94 to 0-61
    indices = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11, 12, 13, 14,
               15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1, 29, 30, 31, 32, 33, 34,
               35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46, 47, 48, 49, -1, 50, 51, -1, 52, -1,
               -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59]
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


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    cx = img.shape[0] * (0.5 + (random.random() * 2 - 1) * translate[0])
    cy = img.shape[1] * (0.5 + (random.random() * 2 - 1) * translate[1])
    s = random.random() * (scale[1] - scale[0]) + scale[0]

    M = cv2.getRotationMatrix2D(angle=a, center=(cy, cx), scale=s)
    imw = cv2.warpAffine(img, M, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    # Return warped points also
    if targets is not None:
        center = np.array([cy, cx]).reshape(1, 2)  # order reversed for opencv
        points = targets[:, 1:5].copy()

        # rotation matrix
        M3 = np.eye(3)
        M3[:2] = M

        # add shear (TODO)
        # Mshear = np.eye(3)
        # M3 = M3 @ Mshear

        # warp points
        n = targets.shape[0]
        xy = np.zeros((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2) - center  # x1y1, x2y2, x1y2, x2y1
        xy = ((xy @ M3.T)[:, :2] + center).reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        i = np.all((xy > 0) & (xy < img.shape[0]), 1)
        xy = xy[i]

        # plot
        # import matplotlib.pyplot as plt
        # plt.imshow(imw)
        # plt.plot(xy[:,[0, 2]], xy[:,[1, 3]], '.')

        targets = targets[i]
        targets[:, 1:5] = xy
        return imw, targets
    else:
        return imw

import glob
import os

import cv2
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

class ImageFolder(Dataset):  # for eval-only
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)


    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect', anti_aliasing=True)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):  # for training
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [
            path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.tif', '.txt') for
            path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect', anti_aliasing=True)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
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
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


class ListDataset_xview(Dataset):  # for training
    #@profile
    def __init__(self, folder_path, img_size=416):
        self.img_files = sorted(glob.glob('%s/*.*' % (folder_path + 'train_images')))
        self.img_shape = (img_size, img_size)
        self.label_files = [path.replace('train_images', 'train_labels').replace('.tif', '.txt') for
                            path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 5000

    #@profile
    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)]
        img = cv2.imread(img_path)

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding

        input_img = resize_square(img, height=self.img_shape[0]) / 255.0  # MUCH faster
        padded_h = max(h,w)
        padded_w = padded_h

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
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
            labels[:, 0] = remap_classes(labels[:, 0])
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


def remap_classes(classes):  # remap xview classes 11-94 to 0-61
    c = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11, 12, 13, 14, 15,
         -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1, 29, 30, 31, 32, 33, 34, 35, 36, 37,
         -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46, 47, 48, 49, 50, 51, 52, -1,
         53, -1, -1, 54, 55, 56, -1, 57, -1, -1, 58, -1, 59, -1, 60, 61]
    return [c[int(x)] for x in classes]

def resize_square(im, height=416, pad_color=(128, 128, 128)):  # resizes a rectangular image to a padded square
    shape = im.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)
    new_shape = [int(shape[0] * ratio), int(shape[1] * ratio)]
    dw = height - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    im = cv2.resize(im, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
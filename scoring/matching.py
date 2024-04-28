"""
Copyright 2018 Defense Innovation Unit Experimental All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import defaultdict
from scoring.rectangle import Rectangle
import numpy as np


class Matching(object):
    """Matching class."""

    def __init__(self, groundtruth_rects, rects):
        """
        Constructs a Matching instance.

        Args:
          groundtruth_rects: a list of groundtruth rectangles.
          rects: a list of rectangles to be matched against the groundtruth_rects.
        Raises:
          ValueError: if any item inside the groundtruth_rects or rects are not
          Rectangle type.
        """
        for rect in groundtruth_rects:
            if not isinstance(rect, Rectangle):
                raise ValueError("Invalid instance type: should be Rectangle.")
        for rect in rects:
            if not isinstance(rect, Rectangle):
                raise ValueError("Invalid instance type: should be Rectangle.")
        self.groundtruth_rects_ = groundtruth_rects
        self.rects_ = rects
        self._compute_iou_from_rectangle_pairs()

    def _compute_iou_from_rectangle_pairs(self):
        """Computes the iou scores between all pairs of rectangles."""
        # try to extract a matrix nx4 from rects
        m = len(self.groundtruth_rects_)
        n = len(self.rects_)
        self.n = n
        self.m = m

        self.iou_rectangle_pair_indices_ = defaultdict(list)

        if not (n == 0 or m == 0):
            mat2 = np.array([j.coords for j in self.groundtruth_rects_])
            mat1 = np.array([j.coords for j in self.rects_])
            # i,j axes correspond to #boxes, #coords per rect

            # compute the areas
            w1 = mat1[:, 2] - mat1[:, 0]
            w2 = mat2[:, 2] - mat2[:, 0]
            h1 = mat1[:, 3] - mat1[:, 1]
            h2 = mat2[:, 3] - mat2[:, 1]
            a1 = np.multiply(h1, w1)
            a2 = np.multiply(h2, w2)
            w_h_matrix = cartesian([a1, a2]).reshape((n, m, 2))
            a_matrix = w_h_matrix.sum(axis=2)

            # now calculate the intersection rectangle
            i_xmin = cartesian([mat1[:, 0], mat2[:, 0]]).reshape((n, m, 2))
            i_xmax = cartesian([mat1[:, 2], mat2[:, 2]]).reshape((n, m, 2))
            i_ymin = cartesian([mat1[:, 1], mat2[:, 1]]).reshape((n, m, 2))
            i_ymax = cartesian([mat1[:, 3], mat2[:, 3]]).reshape((n, m, 2))
            i_w = np.min(i_xmax, axis=2) - np.max(i_xmin, axis=2)
            i_h = np.min(i_ymax, axis=2) - np.max(i_ymin, axis=2)
            i_w[i_w < 0] = 0
            i_h[i_h < 0] = 0

            i_a_matrix = np.multiply(i_w, i_h)
            iou_matrix = np.divide(i_a_matrix, (a_matrix - i_a_matrix))
            self.iou_matrix = iou_matrix

        else:
            self.iou_matrix = np.zeros((n, m))

    def greedy_match(self, iou_threshold):
        """Performs greedy matching of rectangles based on IOU threshold, returning matched indices."""
        gt_rects_matched = [False for gt_index in range(self.m)]
        rects_matched = [False for r_index in range(self.n)]

        if self.n == 0:
            return [], []
        elif self.m == 0:
            return rects_matched, []

        for i, gt_index in enumerate(np.argmax(self.iou_matrix, axis=1)):
            if self.iou_matrix[i, gt_index] >= iou_threshold:
                if gt_rects_matched[gt_index] is False and rects_matched[i] is False:
                    rects_matched[i] = True
                    gt_rects_matched[gt_index] = True
        return rects_matched, gt_rects_matched


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out

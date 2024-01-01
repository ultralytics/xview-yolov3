"""Utility methods for computing the performance metrics."""
from matching import Matching
from rectangle import Rectangle


def safe_divide(numerator, denominator):
    """Computes the safe division to avoid the divide by zero problem."""
    if denominator == 0:
        return 0
    return numerator / denominator


def compute_statistics_given_rectangle_matches(groundtruth_rects_matched, rects_matched):
    """
    Computes the staticstics given the groundtruth_rects and rects matches.

    Args:
        image_id: the image_id referring to the image to be evaluated.
        groundtruth_rects_matched: the groundtruth_rects_matched represents
          a list of integers returned from the Matching class instance to
          indicate the matched rectangle indices from rects for each of the
          groundtruth_rects.
        rects_matched: the rects_matched represents a list of integers returned
          from the Matching class instance to indicate the matched rectangle
          indices from groundtruth_rects for each of the rects.
    Returns:
        A dictionary holding the computed statistics as well as the inputs.
    """
    # Calculate the total_positives, true_positives, and false_positives.
    total_positives = len(groundtruth_rects_matched)
    true_positives = sum(item is not None for item in groundtruth_rects_matched)
    false_positives = sum(item is None for item in rects_matched)
    return {
        "groundtruth_rects_matched": groundtruth_rects_matched,
        "rects_matched": rects_matched,
        "total_positives": total_positives,
        "true_positives": true_positives,
        "false_positives": false_positives,
    }


def compute_precision_recall_given_image_statistics_list(iou_threshold, image_statistics_list):
    """
    Computes the precision recall numbers given iou_threshold and statistics.

    Args:
        iou_threshold: the iou_threshold under which the statistics are computed.
        image_statistics_list: a list of the statistics computed and returned
        by the compute_statistics_given_rectangle_matches method for a list of
        images.
    Returns:
        A dictionary holding the precision, recall as well as the inputs.
    """
    total_positives = 0
    true_positives = 0
    false_positives = 0
    for statistics in image_statistics_list:
        total_positives += statistics["total_positives"]
        true_positives += statistics["true_positives"]
        false_positives += statistics["false_positives"]
    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, total_positives)
    return {
        "iou_threshold": iou_threshold,
        "precision": precision,
        "recall": recall,
        "image_statistics_list": image_statistics_list,
    }


def compute_average_precision_recall_given_precision_recall_dict(precision_recall_dict):
    """
    Computes the average precision (AP) and average recall (AR).

    Args:
        precision_recall_dict: the precision_recall_dict holds the dictionary of
        precision and recall information returned by the
        compute_precision_recall_given_image_statistics_list method, which is
        calculated under a range of iou_thresholds, where the iou_threshold is
        the key.
    Returns:
        average_precision, average_recall.
    """
    precision = 0
    recall = 0
    for _, value in precision_recall_dict.items():
        precision += value["precision"]
        recall += value["recall"]
    average_precision = safe_divide(precision, len(precision_recall_dict))
    average_recall = safe_divide(recall, len(precision_recall_dict))
    return average_precision, average_recall


def convert_to_rectangle_list(coordinates):
    """Converts the coordinates in a list to the Rectangle list."""
    rectangle_list = []
    number_of_rects = int(len(coordinates) / 4)
    for i in range(number_of_rects):
        rectangle_list.append(
            Rectangle(coordinates[4 * i], coordinates[4 * i + 1], coordinates[4 * i + 2], coordinates[4 * i + 3])
        )
    return rectangle_list


def compute_average_precision_recall(groundtruth_coordinates, coordinates, iou_threshold):
    """
    Computes the average precision (AP) and average recall (AR).

    Args:
        groundtruth_info_dict: the groundtruth_info_dict holds all the groundtruth
          information for an evaluation dataset. The format of this groundtruth_info_dict is
          as follows:
          {'image_id_0':
           [xmin_0,ymin_0,xmax_0,ymax_0,...,xmin_N0,ymin_N0,xmax_N0,ymax_N0],
           ...,
           'image_id_M':
           [xmin_0,ymin_0,xmax_0,ymax_0,...,xmin_NM,ymin_NM,xmax_NM,ymax_NM]},
          where
            image_id_* is an image_id that has the groundtruth rectangles labeled.
            xmin_*,ymin_*,xmax_*,ymax_* is the top-left and bottom-right corners
              of one groundtruth rectangle.

        test_info_dict: the test_info_dict holds all the test information for an
          evaluation dataset.
           The format of this test_info_dict is the same
          as the above groundtruth_info_dict.

        iou_threshold_range: the IOU threshold range to compute the average
          precision (AP) and average recall (AR). For example:
          iou_threshold_range = [0.50:0.05:0.95]
    Returns:
        average_precision, average_recall, as well as the precision_recall_dict,
        where precision_recall_dict holds the full precision/recall information
        for each of the iou_threshold in the iou_threshold_range.
    Raises:
        ValueError: if the input groundtruth_info_dict and test_info_dict show
        inconsistent information.
    """

    # Start to build up the Matching instances for each of the image_id_*, which
    # is to hold the IOU computation between the rectangle pairs for the same
    # image_id_*.
    matchings = {}
    if (len(groundtruth_coordinates) % 4 != 0) or (len(coordinates) % 4 != 0):
        raise ValueError("groundtruth_info_dict and test_info_dict should hold " "only 4 * N numbers.")

    groundtruth_rects = convert_to_rectangle_list(groundtruth_coordinates)
    rects = convert_to_rectangle_list(coordinates)
    matching = Matching(groundtruth_rects, rects)

    image_statistics_list = []
    groundtruth_rects_matched, rects_matched = matching.matching_by_greedy_assignment(iou_threshold)

    image_statistics = compute_statistics_given_rectangle_matches(groundtruth_rects_matched, rects_matched)
    image_statistics_list.append(image_statistics)

    # Compute the precision and recall under this iou_threshold.
    precision_recall = compute_precision_recall_given_image_statistics_list(iou_threshold, image_statistics_list)

    # Compute the average_precision and average_recall.
    # average_precision, average_recall = (
    #    compute_average_precision_recall_given_precision_recall_dict(
    #        precision_recall_dict))

    return precision_recall

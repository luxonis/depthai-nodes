from itertools import product
from typing import List, Optional, Tuple

import numpy as np

from depthai_nodes.node.parsers.utils import normalize_bboxes
from depthai_nodes.node.parsers.utils.keypoints import normalize_keypoints


def manual_product(*args):
    """You can use this function instead of itertools.product."""
    if not args:
        return [()]
    result = [[]]
    for pool in args:
        result = [x + [y] for x in result for y in pool]
    return [tuple(x) for x in result]


def generate_anchors(
    input_size: Tuple[int, int],
    min_sizes: Optional[List[List[int]]] = None,
    strides: Optional[List[int]] = None,
):
    """Generate a set of default bounding boxes, known as anchors.
    The code is taken from https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample/tree/main

    @param input_size: A tuple representing the width and height of the input image.
    @type input_size: Tuple[int, int]
    @param min_sizes: A list of lists, where each inner list contains the minimum sizes of the anchors for different feature maps. If None then '[[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]' will be used. Defaults to None.
    @type min_sizes Optional[List[List[int]]]
    @param strides: Strides for each feature map layer. If None then '[8, 16, 32, 64]' will be used. Defaults to None.
    @type strides: Optional[List[int]]
    @return: Anchors.
    @rtype: np.ndarray
    """
    w, h = input_size

    if min_sizes is None:
        min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    if strides is None:
        strides = [8, 16, 32, 64]

    # Calculate sizes of different feature maps by progressively halving the dimensions of the input image.
    feature_map_2th = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
    feature_map_3th = [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)]
    feature_map_4th = [int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)]
    feature_map_5th = [int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)]
    feature_map_6th = [int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2)]
    feature_maps = [feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th]

    # Generate anchors
    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes_k = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size_k in min_sizes_k:
                s_kx = min_size_k / w
                s_ky = min_size_k / h

                cx = (j + 0.5) * strides[k] / w
                cy = (i + 0.5) * strides[k] / h

                anchors.append([cx, cy, s_kx, s_ky])

    anchors = np.array(anchors, dtype=np.float32)
    return anchors


def decode_detections(
    input_size: Tuple[int, int],
    loc: np.ndarray,
    conf: np.ndarray,
    iou: np.ndarray,
    variance: Optional[List[float]] = None,
):
    """
    Decodes the output of an object detection model by converting the model's predictions (localization, confidence, and IoU scores) into bounding boxes, keypoints, and scores.
    The code is taken from https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample/tree/main

    @param input_size: The size of the input image (height, width).
    @type input_size: tuple
    @param loc: The predicted locations (or offsets) of the bounding boxes.
    @type loc: np.ndarray
    @param conf: The predicted class confidence scores.
    @type conf: np.ndarray
    @param iou: The predicted IoU (Intersection over Union) scores.
    @type iou: np.ndarray
    @param variance: A list of variances used to decode the bounding box predictions. If None then [0.1,0.2] will be used. Defaults to None.
    @type variance: Optional[List[float]]
    @return: A tuple of bboxes, keypoints, and scores.
        - bboxes: NumPy array of shape (N, 4) containing the decoded bounding boxes in the format [x_min, y_min, width, height].
        - keypoints: A NumPy array of shape (N, 10) containing the decoded keypoint coordinates for each anchor.
        - scores: A NumPy array of shape (N, 1) containing the combined scores for each anchor.
    @rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    """

    w, h = input_size

    if variance is None:
        variance = [0.1, 0.2]

    anchors = generate_anchors(input_size)

    # Get scores
    cls_scores = conf[:, 1]
    iou_scores = iou[:, 0]
    _idx = np.where(iou_scores < 0.0)
    iou_scores[_idx] = 0.0
    _idx = np.where(iou_scores > 1.0)
    iou_scores[_idx] = 1.0
    scores = np.sqrt(cls_scores * iou_scores)
    scores = scores[:, np.newaxis]

    # Get bounding boxes
    scale = np.array((w, h))
    bboxes = np.hstack(
        (
            (anchors[:, 0:2] + loc[:, 0:2] * variance[0] * anchors[:, 2:4]) * scale,
            (anchors[:, 2:4] * np.exp(loc[:, 2:4] * variance)) * scale,
        )
    )
    bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

    # Get keypoints
    keypoints = np.hstack(
        (
            (anchors[:, 0:2] + loc[:, 4:6] * variance[0] * anchors[:, 2:4]) * scale,
            (anchors[:, 0:2] + loc[:, 6:8] * variance[0] * anchors[:, 2:4]) * scale,
            (anchors[:, 0:2] + loc[:, 8:10] * variance[0] * anchors[:, 2:4]) * scale,
            (anchors[:, 0:2] + loc[:, 10:12] * variance[0] * anchors[:, 2:4]) * scale,
            (anchors[:, 0:2] + loc[:, 12:14] * variance[0] * anchors[:, 2:4]) * scale,
        )
    )

    return bboxes, keypoints, scores


def prune_detections(
    bboxes: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, conf_threshold: float
):
    """Prune detections based on confidence threshold.

    Parameters:
    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes.
    @type np.ndarray
    @param keypoints: A numpy array of shape (N, 10) containing the keypoints.
    @type np.ndarray
    @param scores: A numpy array of shape (N,) containing the scores.
    @type np.ndarray
    @param conf_threshold: The confidence threshold.
    @type float
    @return: A tuple of bboxes, keypoints, and scores.
        - bboxes: NumPy array of shape (N, 4) containing the decoded bounding boxes in the format [x_min, y_min, width, height].
        - keypoints: A NumPy array of shape (N, 10) containing the decoded keypoint coordinates for each anchor.
        - scores: A NumPy array of shape (N, 1) containing the combined scores for each anchor.
    @rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    keep_indices = np.where(scores.squeeze() > conf_threshold)
    return bboxes[keep_indices], keypoints[keep_indices], scores[keep_indices]


def format_detections(
    bboxes: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    input_size: Tuple[int, int],
):
    """Format detections into a list of dictionaries.

    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes.
    @type np.ndarray
    @param keypoints: A numpy array of shape (N, 10) containing the keypoints.
    @type np.ndarray
    @param scores: A numpy array of shape (N,) containing the scores.
    @type np.ndarray
    @param input_size: A tuple representing the width and height of the input image.
    @type input_size: tuple
    @return: A tuple of bboxes, keypoints, and scores.
        - bboxes: NumPy array of shape (N, 4) containing the decoded bounding boxes in the format [x_min, y_min, width, height].
        - keypoints: A NumPy array of shape (N, 10) containing the decoded keypoint coordinates for each anchor.
        - scores: A NumPy array of shape (N, 1) containing the combined scores for each anchor.
    @rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    w, h = input_size

    bboxes = normalize_bboxes(bboxes, height=h, width=w)

    keypoints = keypoints.astype(np.int32)
    keypoints = keypoints.reshape(-1, 2)
    keypoints = normalize_keypoints(keypoints, height=h, width=w)
    keypoints = keypoints.reshape(-1, 5, 2)  # (N,2) to (N,5,2)

    scores = np.array([scores]).flatten()

    return bboxes, keypoints, scores


def decode_and_prune_detections(
    input_size: Tuple[int, int],
    loc: np.ndarray,
    conf: np.ndarray,
    iou: np.ndarray,
    conf_threshold: float,
    anchors: np.ndarray,
    variance: Optional[List[float]] = None,
):
    """Optimized function that combines decode_detections and prune_detections. Performs
    early pruning to avoid processing low-confidence detections.

    @param input_size: The size of the input image (width, height).
    @param loc: The predicted locations (or offsets) of the bounding boxes.
    @param conf: The predicted class confidence scores.
    @param iou: The predicted IoU (Intersection over Union) scores.
    @param conf_threshold: The confidence threshold for pruning.
    @param anchors: Pre-computed anchors to avoid regeneration.
    @param variance: A list of variances used to decode the bounding box predictions.
    @return: A tuple of bboxes, keypoints, and scores.
    """
    w, h = input_size

    if variance is None:
        variance = [0.1, 0.2]

    # Convert variance to numpy arrays for vectorized operations
    var0 = variance[0]
    var1 = variance[1]

    # Get scores and apply early pruning
    cls_scores = conf[:, 1]
    iou_scores = np.clip(iou[:, 0], 0.0, 1.0)  # Clip in one operation
    scores = np.sqrt(cls_scores * iou_scores)

    # Early pruning - only process detections above threshold
    keep_mask = scores > conf_threshold
    if not np.any(keep_mask):
        # Return empty arrays if no detections pass threshold
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 10), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )

    # Filter all inputs early
    loc_filtered = loc[keep_mask]
    anchors_filtered = anchors[keep_mask]
    scores_filtered = scores[keep_mask]

    # Pre-compute scale and anchor components
    scale = np.array([w, h], dtype=np.float32)
    anchor_xy = anchors_filtered[:, 0:2]
    anchor_wh = anchors_filtered[:, 2:4]

    # Compute bounding boxes with vectorized operations
    # Center coordinates
    center_xy = (anchor_xy + loc_filtered[:, 0:2] * var0 * anchor_wh) * scale
    # Width and height
    wh = (anchor_wh * np.exp(loc_filtered[:, 2:4] * var1)) * scale
    # Convert to top-left format
    top_left = center_xy - wh / 2
    bboxes = np.hstack((top_left, wh))

    # Vectorized keypoint computation
    keypoints = np.zeros((len(anchors_filtered), 10), dtype=np.float32)
    for i in range(5):
        start_idx = 4 + i * 2
        end_idx = start_idx + 2
        keypoint_xy = (
            anchor_xy + loc_filtered[:, start_idx:end_idx] * var0 * anchor_wh
        ) * scale
        keypoints[:, i * 2 : (i + 1) * 2] = keypoint_xy

    scores_filtered = scores_filtered[:, np.newaxis]

    return bboxes, keypoints, scores_filtered

import numpy as np
from itertools import product # NOTE: you can use manual_product function instead of itertools.product

from .bbox import normalize_bboxes
from .keypoints import normalize_keypoints


def manual_product(*args):
    """You can use this function instead of itertools.product"""
    if not args:
        return [()]
    result = [[]]
    for pool in args:
        result = [x + [y] for x in result for y in pool]
    return [tuple(x) for x in result]


def generate_anchors(
    input_shape,
    min_sizes=[[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    strides=[8, 16, 32, 64],
):
    """Generate a set of default bounding boxes, known as anchors.
    The code is taken from https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample/tree/main

    @param input_shape: A tuple representing the height and width of the input image.
    @type input_shape: tuple
    @param min_sizes: A list of lists, where each inner list contains the minimum sizes of the anchors for different feature maps.
    @type min_sizes List[List[int]]
    @param strides: Strides for each feature map layer.
    @type strides: List[int]
    @return: Anchors.
    @rtype: np.ndarray
    """
    w, h = input_shape

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
    input_shape,
    loc,
    conf,
    iou,
    variance=[0.1, 0.2],
):
    """
    Decodes the output of an object detection model by converting the model's predictions (localization, confidence, and IoU scores) into bounding boxes, keypoints, and scores.
    The code is taken from https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample/tree/main
    
    @param input_shape: The shape of the input image (height, width).
    @type input_shape: tuple
    @param loc: The predicted locations (or offsets) of the bounding boxes.
    @type loc: np.ndarray
    @param conf: The predicted class confidence scores.
    @type conf: np.ndarray
    @param iou: The predicted IoU (Intersection over Union) scores.
    @type iou: np.ndarray
    @param variance: A list of variances used to decode the bounding box predictions.
    @type variance: List[float]
    @return: A tuple of bboxes, keypoints, and scores.
        - bboxes: NumPy array of shape (N, 4) containing the decoded bounding boxes in the format [x_min, y_min, width, height].
        - keypoints: A NumPy array of shape (N, 10) containing the decoded keypoint coordinates for each anchor.
        - scores: A NumPy array of shape (N, 1) containing the combined scores for each anchor.
    @rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    """

    w,h = input_shape

    anchors = generate_anchors(input_shape)

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
    scale = np.array((w,h))
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


def prune_detections(bboxes, keypoints, scores, conf_threshold):
    """
    Prune detections based on confidence threshold.

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


def format_detections(bboxes, keypoints, scores, input_shape):
    """
    Format detections into a list of dictionaries.

    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes.
    @type np.ndarray
    @param keypoints: A numpy array of shape (N, 10) containing the keypoints.
    @type np.ndarray
    @param scores: A numpy array of shape (N,) containing the scores.
    @type np.ndarray
    @param input_shape: A tuple representing the height and width of the input image.
    @type input_shape: tuple
    @return: A tuple of bboxes, keypoints, and scores.
        - bboxes: NumPy array of shape (N, 4) containing the decoded bounding boxes in the format [x_min, y_min, width, height].
        - keypoints: A NumPy array of shape (N, 10) containing the decoded keypoint coordinates for each anchor.
        - scores: A NumPy array of shape (N, 1) containing the combined scores for each anchor.
    @rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    w, h = input_shape

    bboxes = normalize_bboxes(bboxes, height=h, width=w)

    keypoints = keypoints.astype(np.int32)
    keypoints = keypoints.reshape(-1, 5, 2) # (N,10) to (N,5,2)
    keypoints = normalize_keypoints(keypoints, height=h, width=w)
    
    scores = scores.squeeze()
    
    return bboxes, keypoints, scores

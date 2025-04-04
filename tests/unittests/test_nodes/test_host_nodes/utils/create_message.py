from typing import List, Union

import depthai as dai
import numpy as np

from depthai_nodes import ImgDetectionExtended
from depthai_nodes.message.creators import (
    create_classification_message,
    create_classification_sequence_message,
    create_cluster_message,
    create_detection_message,
    create_image_message,
    create_keypoints_message,
    create_line_detection_message,
    create_map_message,
    create_regression_message,
    create_segmentation_message,  # TODO: test?
)

LABELS = [0, 1, 2, 3, 4]
LABEL_NAMES = ["a", "b", "c", "d", "e"]
SCORES = [0.2, 0.4, 0.6, 0.8, 1.0]
BBOXES = np.array(
    [
        [0.00, 0.20, 0.00, 0.20],
        [0.20, 0.40, 0.20, 0.40],
        [0.40, 0.60, 0.40, 0.60],
        [0.60, 0.80, 0.60, 0.80],
        [0.80, 1.00, 0.80, 1.00],
    ]
)
ANGLES = [0.0, 0.1, 0.0, 0.3, 0.0]
KEYPOINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
    ]
)
KEYPOINTS_SCORES = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
    ]
)
LINES = np.array(
    [
        [0.0, 0.0, 0.1, 0.1],
        [0.2, 0.2, 0.3, 0.3],
        [0.4, 0.4, 0.5, 0.5],
        [0.6, 0.6, 0.7, 0.7],
        [0.8, 0.8, 0.9, 0.9],
    ]
)
CLUSTERS = [[[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]]
PREDICTIONS = [0.2, 0.3, 0.5]
HEIGHT, WIDTH = 5, 5
MAX_VALUE = 255
ARR_2D = np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH), dtype=np.int16)
ARR_2D_FLOAT = ARR_2D.astype(np.float32)
IMG = np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH, 3), dtype=np.uint8)
MASKS = np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH), dtype=np.uint8)


def create_classifications(
    classes: List[str] = LABEL_NAMES, scores: List[float] = SCORES
):
    return create_classification_message(scores, classes)


def create_classifications_sequence(
    classes: List[str] = LABEL_NAMES,
    scores: List[float] = SCORES,
):
    return create_classification_sequence_message(scores, classes)


def create_clusters(clusters: List[List[List[Union[float, int]]]] = CLUSTERS):
    return create_cluster_message(clusters)


def create_img_detections_extended(
    bboxes: np.ndarray = BBOXES,
    scores: np.ndarray = SCORES,
    angles: np.ndarray = ANGLES,
    labels: np.ndarray = LABELS,
    keypoints: np.ndarray = KEYPOINTS,
    keypoints_scores: np.ndarray = KEYPOINTS_SCORES,
    masks: np.ndarray = MASKS,
):
    return create_detection_message(
        bboxes,
        scores,
        angles,
        labels,
        keypoints,
        keypoints_scores,
        masks,
    )


def create_img_frame(img: np.ndarray = IMG, dtype=dai.ImgFrame.Type.BGR888p):
    return create_image_message(img, dtype)


def create_keypoints(
    keypoints: np.ndarray = KEYPOINTS,
    scores: np.ndarray = KEYPOINTS_SCORES,
):
    return create_keypoints_message(keypoints, scores)


def create_lines(
    lines: np.ndarray = LINES,
    scores: np.ndarray = SCORES,
):
    return create_line_detection_message(lines, scores)


def create_map(map: np.ndarray = ARR_2D_FLOAT):
    return create_map_message(map)


def create_regression(predictions: List[float] = PREDICTIONS):
    return create_regression_message(predictions)


def create_segmentation(mask: np.ndarray = ARR_2D):
    return create_segmentation_message(mask)


def create_img_detection(
    bbox: np.ndarray = BBOXES[0], label: int = LABELS[0], score: float = SCORES[0]
):
    """Creates a dai.ImgDetection object.

    @param det: Detection dict with keys "bbox" ([xmin, ymin, xmax, ymax]), "label"
        (int), and "confidence" (float).
    @type det: dict
    @return: The created dai.ImgDetection object.
    @rtype: dai.ImgDetection
    """
    img_det = dai.ImgDetection()
    img_det.xmin, img_det.ymin, img_det.xmax, img_det.ymax = bbox
    img_det.label = label
    img_det.confidence = score
    return img_det


def create_img_detections(
    bboxs: np.ndarray = BBOXES, labels: List = LABELS, scores: List = SCORES
):
    """Creates a dai.ImgDetections object.

    @param dets: List of detection dicts, each containing "bbox" ([xmin, ymin, xmax,
        ymax]), "label" (int), and "confidence" (float).
    @type dets: List[Dict]
    @return: The created dai.ImgDetections object.
    @rtype: dai.ImgDetections
    """
    img_dets = dai.ImgDetections()
    img_dets.detections = [
        create_img_detection(bbox, label, score)
        for bbox, label, score in zip(bboxs, labels, scores)
    ]
    return img_dets


def create_img_detection_extended(
    bbox: np.ndarray = BBOXES[0], label: int = LABELS[0], score: float = SCORES[0]
):
    """Creates a ImgDetectionExtended object.

    @param det: Detection dict with keys "bbox" ([xmin, ymin, xmax, ymax]), "label"
        (int), and "confidence" (float).
    @type det: dict
    @return: The created ImgDetectionExtended object.
    @rtype: ImgDetectionExtended
    """
    img_det_ext = ImgDetectionExtended()
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    img_det_ext.label = label
    img_det_ext.confidence = score
    img_det_ext.rotated_rect = (x_center, y_center, width, height, 0)
    return img_det_ext

from typing import List, Union

import depthai as dai
import numpy as np

from depthai_nodes import ImgDetectionExtended
import depthai_nodes.message.creators as creators

CLASSES = ["class1", "class2", "class3"]
SCORES = [0.0, 0.25, 0.75]

CLASSIFICATION = {
    "classes": CLASSES,
    "scores": SCORES,
    "sequence_num": 3,
}

COLLECTIONS = {
    "clusters": [
        [[0.0, 0.0], [0.1, 0.1]],
        [[0.2, 0.2], [0.3, 0.3]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
    ],  # two 2-point clusters, and one 3-point cluster
    "lines": [
        [0.0, 0.0, 0.1, 0.1],
        [0.2, 0.2, 0.3, 0.3],
        [0.4, 0.4, 0.5, 0.5],
    ],  # three lines
}

DETECTIONS = {
    "bboxes": [
        [0.00, 0.20, 0.00, 0.20],
        [0.20, 0.40, 0.20, 0.40],
        [0.40, 0.60, 0.40, 0.60],
    ],  # three bboxes
    "angles": [0.0, 0.25, 0.75],
    "labels": [i for i in range(len(CLASSES))],
    "scores": SCORES,
    "keypoints": [
        [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]],
        [[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
        [[0.6, 0.6], [0.7, 0.7], [0.8, 0.8]],
    ],  # three keypoints for each bbox detection
    "keypoints_scores": [
        [0.0, 0.1, 0.2],
        [0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8],
    ],
}

HEIGHT, WIDTH = 5, 5
MAX_VALUE = 50
ARRAYS = {
    "2d": np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH)),  # e.g. mask
    "3d": np.random.randint(0, MAX_VALUE, (HEIGHT, WIDTH, 3)),  # e.g. image
}


def create_classifications(
    classes: List[str] = CLASSIFICATION["classes"],
    scores: List[float] = CLASSIFICATION["scores"],
):
    return creators.create_classification_message(classes=classes, scores=scores)


def create_classifications_sequence(
    classes: List[str] = CLASSIFICATION["classes"],
    scores: List[float] = [
        CLASSIFICATION["scores"],
    ]
    * CLASSIFICATION["sequence_num"],
):
    return creators.create_classification_sequence_message(
        classes=classes, scores=scores
    )


def create_clusters(
    clusters: List[List[List[Union[float, int]]]] = COLLECTIONS["clusters"],
):
    return creators.create_cluster_message(clusters=clusters)


def create_img_frame(
    image: np.ndarray = ARRAYS["3d"].astype(np.uint8),
    img_frame_type=dai.ImgFrame.Type.BGR888p,
):
    return creators.create_image_message(image=image, img_frame_type=img_frame_type)


def create_keypoints(
    keypoints: List[List[float]] = DETECTIONS["keypoints"][0],
    scores: List[float] = DETECTIONS["keypoints_scores"][0],
):
    return creators.create_keypoints_message(keypoints=keypoints, scores=scores)


def create_lines(
    lines: np.ndarray = np.array(COLLECTIONS["lines"]),
    scores: np.ndarray = np.array(SCORES),
):
    return creators.create_line_detection_message(lines=lines, scores=scores)


def create_map(map: np.ndarray = ARRAYS["2d"].astype(np.float32)):
    return creators.create_map_message(map=map)


def create_regression(predictions: List[float] = SCORES):
    return creators.create_regression_message(predictions=predictions)


def create_segmentation(mask: np.ndarray = ARRAYS["2d"].astype(np.int16)):
    return creators.create_segmentation_message(mask=mask)


def create_img_detection(
    bbox: List[float] = DETECTIONS["bboxes"][0],
    label: int = DETECTIONS["labels"][0],
    score: float = DETECTIONS["scores"][0],
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
    bboxs: List[List[float]] = DETECTIONS["bboxes"],
    labels: List[int] = DETECTIONS["labels"],
    scores: List[float] = DETECTIONS["scores"],
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
    bbox: List[float] = DETECTIONS["bboxes"][0],
    label: int = DETECTIONS["labels"][0],
    score: float = DETECTIONS["scores"][0],
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


def create_img_detections_extended(
    bboxes: np.ndarray = np.array(DETECTIONS["bboxes"]),
    scores: np.ndarray = np.array(DETECTIONS["scores"]),
    angles: np.ndarray = np.array(DETECTIONS["angles"]),
    labels: np.ndarray = np.array(DETECTIONS["labels"]),
    keypoints: np.ndarray = np.array(DETECTIONS["keypoints"]),
    keypoints_scores: np.ndarray = np.array(DETECTIONS["keypoints_scores"]),
    masks: np.ndarray = ARRAYS["2d"].astype(np.int16),
):
    return creators.create_detection_message(
        bboxes=bboxes,
        scores=scores,
        angles=angles,
        labels=labels,
        keypoints=keypoints,
        keypoints_scores=keypoints_scores,
        masks=masks,
    )

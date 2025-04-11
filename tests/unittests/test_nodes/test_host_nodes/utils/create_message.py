from typing import List, Union

import depthai as dai
import numpy as np

import depthai_nodes.message.creators as creators
from depthai_nodes import ImgDetectionExtended

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
    "lines": np.array(
        [
            [0.0, 0.0, 0.1, 0.1],
            [0.2, 0.2, 0.3, 0.3],
            [0.4, 0.4, 0.5, 0.5],
        ]
    ),  # three lines
}

DETECTIONS = {
    "bboxes": np.array(
        [
            [0.00, 0.20, 0.00, 0.20],
            [0.20, 0.40, 0.20, 0.40],
            [0.40, 0.60, 0.40, 0.60],
        ]
    ),  # three bboxes
    "angles": np.array([0.0, 0.25, 0.75]),
    "labels": np.array([i for i in range(len(CLASSES))]),
    "scores": np.array(SCORES),
    "keypoints": np.array(
        [
            [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]],
            [[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
            [[0.6, 0.6], [0.7, 0.7], [0.8, 0.8]],
        ]
    ),  # three keypoints for each bbox detection
    "keypoints_scores": np.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
        ]
    ),
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
    image: np.ndarray = ARRAYS["3d"],
    img_frame_type=dai.ImgFrame.Type.BGR888p,
):
    return creators.create_image_message(
        image=image.astype(np.uint8), img_frame_type=img_frame_type
    )


def create_keypoints(
    keypoints: np.ndarray = DETECTIONS["keypoints"][0],
    scores: np.ndarray = DETECTIONS["keypoints_scores"][0],
):
    return creators.create_keypoints_message(keypoints=keypoints, scores=scores)


def create_lines(
    lines: np.ndarray = COLLECTIONS["lines"],
    scores: List[float] = SCORES,
):
    return creators.create_line_detection_message(
        lines=np.array(lines), scores=np.array(scores)
    )


def create_map(map: np.ndarray = ARRAYS["2d"]):
    return creators.create_map_message(map=map.astype(np.float32))


def create_regression(predictions: List[float] = SCORES):
    return creators.create_regression_message(predictions=predictions)


def create_segmentation(mask: np.ndarray = ARRAYS["2d"]):
    return creators.create_segmentation_message(mask=mask.astype(np.int16))


def create_img_detection(
    bbox: np.ndarray = DETECTIONS["bboxes"][0],
    label: np.ndarray = DETECTIONS["labels"][0],
    score: np.ndarray = DETECTIONS["scores"][0],
):
    """Creates a dai.ImgDetection object.

    @param det: Detection dict with keys "bbox" ([xmin, ymin, xmax, ymax]), "label"
        (int), and "confidence" (float).
    @type det: dict
    @return: The created dai.ImgDetection object.
    @rtype: dai.ImgDetection
    """
    img_det = dai.ImgDetection()
    img_det.xmin, img_det.ymin, img_det.xmax, img_det.ymax = bbox.tolist()
    img_det.label = label.item()
    img_det.confidence = score.item()
    return img_det


def create_img_detections(
    bboxs: np.ndarray = DETECTIONS["bboxes"],
    labels: np.ndarray = DETECTIONS["labels"],
    scores: np.ndarray = DETECTIONS["scores"],
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
    bbox: np.ndarray = DETECTIONS["bboxes"][0],
    label: np.ndarray = DETECTIONS["labels"][0],
    score: np.ndarray = DETECTIONS["scores"][0],
):
    """Creates a ImgDetectionExtended object.

    @param det: Detection dict with keys "bbox" ([xmin, ymin, xmax, ymax]), "label"
        (int), and "confidence" (float).
    @type det: dict
    @return: The created ImgDetectionExtended object.
    @rtype: ImgDetectionExtended
    """
    img_det_ext = ImgDetectionExtended()
    xmin, ymin, xmax, ymax = bbox.tolist()
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    img_det_ext.label = label.item()
    img_det_ext.confidence = score.item()
    img_det_ext.rotated_rect = (x_center, y_center, width, height, 0)
    return img_det_ext


def create_img_detections_extended(
    bboxes: np.ndarray = DETECTIONS["bboxes"],
    scores: np.ndarray = DETECTIONS["scores"],
    angles: np.ndarray = DETECTIONS["angles"],
    labels: np.ndarray = DETECTIONS["labels"],
    keypoints: np.ndarray = DETECTIONS["keypoints"],
    keypoints_scores: np.ndarray = DETECTIONS["keypoints_scores"],
    masks: np.ndarray = ARRAYS["2d"],
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

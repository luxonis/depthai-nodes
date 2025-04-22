from datetime import timedelta
from typing import List

import depthai as dai
import numpy as np

import depthai_nodes.message.creators as creators
from depthai_nodes import ImgDetectionExtended

from .constants import ARRAYS, DETECTIONS


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
    bboxs: np.ndarray = DETECTIONS["bboxes"],
    labels: np.ndarray = DETECTIONS["labels"],
    scores: np.ndarray = DETECTIONS["scores"],
    timestamp: int = timedelta(days=1, hours=1, minutes=1, seconds=1, milliseconds=0),
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
        create_img_detection(bbox.tolist(), label.item(), score.item())
        for bbox, label, score in zip(bboxs, labels, scores)
    ]
    img_dets.setTimestamp(timestamp)
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

from datetime import timedelta

import depthai as dai
import numpy as np

from .constants import DETECTIONS


def create_img_detection(
    bbox: list[float] = DETECTIONS["bboxes"][0],
    label: int = DETECTIONS["labels"][0],
    score: float = DETECTIONS["scores"][0],
):
    """Create a single detection for tests.

    Args:
        bbox: Normalized ``[xmin, ymin, xmax, ymax]`` coordinates.
        label: Class index.
        score: Detection confidence.

    Returns:
        ``dai.ImgDetection``: Detection populated with the supplied values.
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
    """Create a timestamped detection message for tests.

    Args:
        bboxs: Array of normalized ``[xmin, ymin, xmax, ymax]`` boxes.
        labels: Class indexes corresponding to the boxes.
        scores: Confidence scores corresponding to the boxes.
        timestamp: Timestamp assigned to the message.

    Returns:
        ``dai.ImgDetections``: Message containing the supplied detections.
    """
    img_dets = dai.ImgDetections()
    img_dets.detections = [
        create_img_detection(bbox.tolist(), label.item(), score.item())
        for bbox, label, score in zip(bboxs, labels, scores)
    ]
    img_dets.setTimestamp(timestamp)
    return img_dets

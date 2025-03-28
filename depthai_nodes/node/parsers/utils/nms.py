from typing import List

import cv2
import numpy as np


def nms(dets: np.ndarray, nms_thresh: float = 0.5) -> List[int]:
    """Non-maximum suppression.

    @param dets: Bounding boxes and confidence scores.
    @type dets: np.ndarray
    @param nms_thresh: Non-maximum suppression threshold.
    @type nms_thresh: float
    @return: Indices of the detections to keep.
    @rtype: List[int]
    """
    thresh = nms_thresh
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_cv2(
    bboxes: np.ndarray,
    scores: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
):
    """Non-maximum suppression from the opencv-python library.

    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes.
    @type bboxes: np.ndarray
    @param scores: A numpy array of shape (N,) containing the scores.
    @type scores: np.ndarray
    @param nms_thresh: Non-maximum suppression threshold.
    @type nms_thresh: float
    @return: Indices of the detections to keep.
    @rtype: List[int]
    """

    # NMS
    if len(bboxes) == 0 or len(scores) == 0:
        return []
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=bboxes.tolist(),
        scores=scores.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
        top_k=max_det,
    )

    return keep_indices

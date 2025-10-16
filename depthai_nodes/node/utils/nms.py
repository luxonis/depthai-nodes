from typing import List, Union

import depthai as dai
import numpy as np

from depthai_nodes.message.img_detections import ImgDetectionExtended


def nms_detections(
    detections: List[Union[dai.ImgDetection, ImgDetectionExtended]],
    conf_thresh=0.3,
    iou_thresh=0.4,
):
    """Applies Non-Maximum Suppression (NMS) on a list of dai.ImgDetection objects.

    @param detections: List of dai.ImgDetection objects. @type
    detections: list[dai.ImgDetection] @param conf_thresh: Confidence
    threshold for filtering boxes. @type conf_thresh: float @param
    iou_thresh: IoU threshold for Non-Maximum Suppression (NMS). @type
    iou_thresh: float

    @return: A list of dai.ImgDetection objects after applying NMS.
    @rtype: list[dai.ImgDetection]
    """
    # Filter out detections below confidence threshold
    filtered_detections = [det for det in detections if det.confidence >= conf_thresh]
    if len(filtered_detections) == 0:
        return []

    # Organize detections by class
    detections_by_class = {}
    for det in filtered_detections:
        label = det.label
        if label not in detections_by_class:
            detections_by_class[label] = []
        detections_by_class[label].append(det)

    final_detections = []
    for _, dets in detections_by_class.items():
        boxes = []
        scores = []
        for det in dets:
            # Coordinates are normalized between 0 and 1
            if isinstance(det, dai.ImgDetection):
                boxes.append([det.xmin, det.ymin, det.xmax, det.ymax])
            elif isinstance(det, ImgDetectionExtended):
                boxes.append(det.rotated_rect.getOuterRect())
            scores.append(det.confidence)

        boxes = np.array(boxes)
        scores = np.array(scores)

        # Perform NMS
        keep_indices = nms(boxes, scores, iou_thresh)

        # Keep the detections after NMS
        final_dets = [dets[i] for i in keep_indices]
        final_detections.extend(final_dets)

    return final_detections


def nms(boxes, scores, iou_thresh):
    """Perform Non-Maximum Suppression (NMS).

    @param boxes: An ndarray of shape (N, 4), where each row is [xmin, ymin, xmax,
        ymax].
    @type boxes: np.ndarray
    @param scores: An ndarray of shape (N,), containing the confidence scores for each
        box.
    @type scores: np.ndarray
    @param iou_thresh: The IoU threshold for Non-Maximum Suppression (NMS).
    @type iou_thresh: float
    @return: A list of indices of the boxes to keep after applying NMS.
    @rtype: list[int]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # Compute area of each box
    areas = (x2 - x1) * (y2 - y1)
    # Sort the boxes by scores in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = (
            areas[i] + areas[order[1:]] - inter + 1e-6
        )  # Add a small epsilon to prevent division by zero
        iou = inter / union

        # Keep boxes with IoU less than the threshold
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

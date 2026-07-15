import numpy as np

from .bbox_format_converters import xyxy_to_xywh
from .nms import nms_cv2


def compute_detection_outputs(
    bboxes: np.ndarray,
    scores: np.ndarray,
    *,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter detection outputs and convert boxes to center-width-height format."""
    indices = nms_cv2(bboxes, scores, conf_threshold, iou_threshold, max_det)

    if len(indices) == 0:
        return np.array([]), np.array([])

    filtered_bboxes = xyxy_to_xywh(bboxes[indices])
    filtered_scores = scores[indices]
    return filtered_bboxes, filtered_scores

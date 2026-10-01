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
    """Suppress overlapping detections and convert retained boxes.

    Args:
        bboxes: Bounding boxes of shape ``(N, 4)`` in ``[xmin, ymin, xmax, ymax]``
            format.
        scores: Confidence scores of shape ``(N,)``.
        conf_threshold: Minimum detection confidence used to filter candidates.
        iou_threshold: Intersection-over-union threshold for non-maximum suppression.
        max_det: Maximum number of detection candidates to retain or consider during
            suppression.

    Returns:
        Retained center-XY/width/height boxes and corresponding scores. Coordinates
        retain their input units. Both arrays are empty if no boxes survive.
    """
    nms_bboxes = np.column_stack((bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]))
    indices = np.asarray(
        nms_cv2(nms_bboxes, scores, conf_threshold, iou_threshold, max_det)
    ).reshape(-1)

    if indices.size == 0:
        return np.array([]), np.array([])

    filtered_bboxes = xyxy_to_xywh(bboxes[indices])
    filtered_scores = scores[indices]
    return filtered_bboxes, filtered_scores

import cv2
import numpy as np

from depthai_nodes.node.parsers.utils import (
    corners_to_rotated_bbox,
)


def _get_mini_boxes(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Internal function to get the minimum bounding box of a contour.

    Args:
        contour: The contour to get the minimum bounding box of the text.

    Returns:
        The minimum rotated bounding box defined as [x_center, y_center, width, height,
        angle] and the corners of the box.
    """
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0

    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    corners = [points[index_1], points[index_2], points[index_3], points[index_4]]

    bbox = corners_to_rotated_bbox(np.array(corners))
    return bbox, np.array(corners)


def _dilate_box(corners: np.ndarray, pixels: int) -> np.ndarray:
    """Internal function to dilate the bounding box area by a specified number of
    pixels.

    Args:
        corners: The corners of the bounding box.
        pixels: The number of pixels to dilate the bounding box area by.

    Returns:
        The dilated bounding box corners.
    """
    corners[0] = corners[0] - pixels
    corners[1][0] = corners[1][0] + pixels
    corners[1][1] = corners[1][1] - pixels
    corners[2] = corners[2] + pixels
    corners[3][0] = corners[3][0] - pixels
    corners[3][1] = corners[3][1] + pixels

    return corners


def _box_score(predictions: np.ndarray, _corners: np.ndarray) -> float:
    """Internal function to calculate the score of a bounding box based on the mean
    pixel values within the box area.

    Args:
        predictions: The predictions from the model.
        _corners: The corners of the bounding box.

    Returns:
        The score of the bounding box.
    """
    h, w = predictions.shape[:2]
    corners = _dilate_box(_corners, -2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.reshape(-1, 1, 2).astype(np.int32)], 1)

    return float(np.sum(predictions * mask) / (np.sum(mask) + 1e-5))


def _unclip(
    box: np.ndarray,
    unclip_ratio: float = 2,
) -> np.ndarray:
    """Internal function to dilate the bounding box area by a specified ratio.

    Args:
        box: The rotated bounding box.
        unclip_ratio: The ratio to dilate the bounding box area by.

    Returns:
        The dilated bounding box corners.
    """

    box[2] = box[2] * np.sqrt(unclip_ratio)
    box[3] = box[3] * np.sqrt(unclip_ratio)

    return box


def parse_paddle_detection_outputs(
    predictions: np.ndarray,
    mask_threshold: float = 0.25,
    bbox_threshold: float = 0.5,
    max_detections: int = 100,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the output of a PaddlePaddle Text Detection model from a mask of text
    probabilities into rotated bounding boxes with additional corners saved as
    keypoints.

    Args:
        predictions: The output of a PaddlePaddle Text Detection model.
        mask_threshold: The threshold for the mask.
        bbox_threshold: The threshold for bounding boxes.
        max_detections: The maximum number of candidate bounding boxes.
        width: The width of the image.
        height: The height of the image.

    Returns:
        A touple containing the rotated bounding boxes, corners and scores.
    """

    if len(predictions.shape) == 4:
        if predictions.shape[0] == 1 and predictions.shape[1] == 1:
            predictions = predictions[0, 0]
        elif predictions.shape[0] == 1 and predictions.shape[3] == 1:
            predictions = predictions[0, :, :, 0]
        else:
            raise ValueError(
                f"Predictions should be either (1, 1, H, W) or (1, H, W, 1), got {predictions.shape}."
            )
    else:
        raise ValueError(
            f"Predictions should be 4D array of shape (1, 1, H, W) or (1, H, W, 1), got {predictions.shape}."
        )

    if width is None or height is None:
        height, width = predictions.shape[:2]

    mask = (predictions > mask_threshold).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Pre-filter small contours (skip sorting all small blobs)
    contours = [c for c in contours if cv2.contourArea(c) > 8]

    # If too many, pick largest by area only
    if len(contours) > max_detections:
        contour_areas = np.array([cv2.contourArea(c) for c in contours])
        top_indices = np.argpartition(-contour_areas, max_detections)[:max_detections]
        contours = [contours[i] for i in top_indices]

    boxes, angles, scores = [], [], []
    for contour in contours:
        box, corners = _get_mini_boxes(contour)
        if min(box[2], box[3]) < 8:  # Skip tiny boxes
            continue
        score = _box_score(predictions, corners.reshape(-1, 2))
        if score < bbox_threshold:
            continue
        box = _unclip(box)
        boxes.append([box[0] / width, box[1] / height, box[2] / width, box[3] / height])
        angles.append(round(box[4], 0))
        scores.append(score)

    boxes = np.array(boxes, dtype=np.float32)
    angles = np.array(angles, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    if boxes.size > 0:
        boxes = np.clip(boxes, 0.0, 1.0)
    return boxes, angles, scores


def compute_pp_text_detections(
    predictions: np.ndarray,
    *,
    mask_threshold: float,
    conf_threshold: float,
    max_det: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode PaddleOCR text detection output into boxes, angles, and scores."""
    return parse_paddle_detection_outputs(
        predictions,
        mask_threshold,
        conf_threshold,
        max_det,
    )

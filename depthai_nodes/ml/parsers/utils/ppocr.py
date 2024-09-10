from typing import Tuple

import cv2
import numpy as np


def _get_mini_boxes(contour: np.ndarray) -> Tuple[np.ndarray, float]:
    """ Internal function to get the minimum bounding box of a contour.

    @param contour: The contour to get the minimum bounding box of.
    @type contour: np.ndarray
    @return: The minimum bounding box, indexed as [top-left, top-right, bottom-right,
        bottom-left], and the minimum side length.
    @rtype: Tuple[np.ndarray, float]
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

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return np.array(box), min(bounding_box[1])


def _box_score(predictions: np.ndarray, _box: np.ndarray) -> float:
    """Internal function to calculate the score of a bounding box based on the mean
    pixel values within the box area.

    @params predictions: The predictions from the model.
    @type predictions: np.ndarray
    @params _box: The bounding box.
    @type _box: np.ndarray
    @return: The score of the bounding box.
    @rtype: float
    """
    h, w = predictions.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)

    return cv2.mean(predictions[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def _unclip(
    box: np.ndarray, width: int, height: int, unclip_ratio: float = 2
) -> np.ndarray:
    """Internal function to dilate the bounding box area by a specified ratio.

    @param box: The bounding box to dilate.
    @type box: np.ndarray
    @param width: The width of the model output predictions.
    @type width: int
    @param height: The height of the model output predictions.
    @type height: int
    @param unclip_ratio: The ratio to dilate the bounding box area by.
    @type unclip_ratio: float = 2
    @return: The dilated bounding box.
    @rtype: np.ndarray
    """

    perimiter = cv2.arcLength(box, True)
    area = cv2.contourArea(box)
    dilation_pixels = (
        int(-perimiter / 8 + np.sqrt(perimiter**2 / 64 + area * unclip_ratio / 4)) + 1
    )

    box[0] = box[0] - dilation_pixels
    box[1][0] = box[1][0] + dilation_pixels
    box[1][1] = box[1][1] - dilation_pixels
    box[2] = box[2] + dilation_pixels
    box[3][0] = box[3][0] - dilation_pixels
    box[3][1] = box[3][1] + dilation_pixels

    for point in box:
        point[0] = min(max(point[0], 0), width - 1)
        point[1] = min(max(point[1], 0), height - 1)

    return np.array(box, dtype=np.int32)


def parse_paddle_detection_outputs(
    predictions: np.ndarray,
    mask_threshold: float = 0.3,
    bbox_threshold: float = 0.7,
    max_detections: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse all outputs from a PaddlePaddle Text Detection model.

    @param predictions: The output of a PaddlePaddle Text Detection model.
    @type predictions: np.ndarray
    @param mask_threshold: The threshold for the mask.
    @type mask_threshold: float = 0.3
    @param bbox_threshold: The threshold for bounding boxes.
    @type bbox_threshold: float = 0.7
    @param max_detections: The maximum number of candidate bounding boxes.
    @type max_detections: int = 1000
    @return: A touple containing the bounding boxes and scores.
    @rtype: Touple[np.ndarray, np.ndarray]
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

    mask = predictions > mask_threshold
    src_h, src_w = predictions.shape[:2]

    outs = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(outs) == 3:
        _, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    num_contours = min(len(contours), max_detections)

    boxes = []
    scores = []
    for contour in contours[:num_contours]:
        box, sside = _get_mini_boxes(contour)
        if sside < 5:
            continue

        score = _box_score(predictions, box.reshape(-1, 2))
        if score < bbox_threshold:
            continue

        box = _unclip(box, src_w, src_h)

        boxes.append(box.astype(np.int32))
        scores.append(score)

    return np.array(boxes, dtype=np.int32), np.array(scores)


def corners2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from corner to [x_min, y_min, x_max, y_max] format.

    @param boxes: Boxes in corner format.
    @type boxes: np.ndarray of shape (n, 4, 2)
    @return: Boxes in [x_min, y_min, x_max, y_max] format.
    @rtype: np.ndarray
    """

    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    if len(boxes.shape) != 3:
        raise ValueError(
            f"Boxes should be 3D array of shape (n, 4, 2), got {boxes.shape}."
        )

    if boxes.shape[1] != 4 or boxes.shape[2] != 2:
        raise ValueError(f"Each box should be of shape (4, 2), got {boxes.shape[1:]}")

    mins = boxes[:, 0, :]
    maxs = boxes[:, 2, :]

    return np.concatenate([mins, maxs], axis=1)

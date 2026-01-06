from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from depthai_nodes.node.parsers.utils import sigmoid
from depthai_nodes.node.parsers.utils.yolo import (
    decode_yolo_output,
)


def box_prompt(
    masks: np.ndarray, bbox: Tuple[int, int, int, int], orig_shape: Tuple[int, int]
) -> np.ndarray:
    """Modifies the bounding box properties and calculates IoU between masks and
    bounding box.

    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/prompt.py#L286
    Modified so it uses numpy instead of torch.

    @param masks: The resulting masks of the FastSAM model
    @type masks: np.ndarray
    @param bbox: The prompt bounding box coordinates
    @type bbox: Tuple[int, int, int, int]
    @param orig_shape: The original shape of the image
    @type orig_shape: Tuple[int, int] (height, width)
    @return: The modified masks
    @rtype: np.ndarray
    """
    if masks is not None:
        assert bbox[2] != 0 and bbox[3] != 0
        target_height, target_width = orig_shape
        h, w = masks.shape[1:3]
        if h != target_height or w != target_width:
            bbox = [
                int(bbox[0] * w / target_width),
                int(bbox[1] * h / target_height),
                int(bbox[2] * w / target_width),
                int(bbox[3] * h / target_height),
            ]
        bbox[0] = max(round(bbox[0]), 0)
        bbox[1] = max(round(bbox[1]), 0)
        bbox[2] = min(round(bbox[2]), w)
        bbox[3] = min(round(bbox[3]), h)

        bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

        masks_area = np.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], axis=(1, 2))
        orig_masks_area = np.sum(masks, axis=(1, 2))

        union = bbox_area + orig_masks_area - masks_area
        iou = masks_area / union
        max_iou_index = np.argmax(iou)

        masks = masks[max_iou_index]
    return masks.reshape(1, masks.shape[0], masks.shape[1])


def format_results(
    bboxes: np.ndarray, masks: np.ndarray, filter: int = 0
) -> List[Dict[str, Any]]:
    """Formats detection results into list of annotations each containing ID,
    segmentation, bounding box, score and area.

    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/prompt.py#L56

    @param bboxes: The bounding boxes of the detected objects
    @type bboxes: np.ndarray
    @param masks: The masks of the detected objects
    @type masks: np.ndarray
    @param filter: The filter value
    @type filter: int
    @return: The formatted annotations
    @rtype: List[Dict[str, Any]]
    """
    annotations = []
    n = len(masks) if masks is not None else 0
    for i in range(n):
        mask = masks[i] == 1.0
        if np.sum(mask) >= filter:
            annotation = {
                "id": i,
                "segmentation": mask,
                "bbox": bboxes[i][:4],
                "score": bboxes[i][4],
            }
            annotation["area"] = annotation["segmentation"].sum()
            annotations.append(annotation)
    return annotations


def point_prompt(
    bboxes: np.ndarray,
    masks: np.ndarray,
    points: List[Tuple[int, int]],
    pointlabel: List[int],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """Adjusts points on detected masks based on user input and returns the modified
    results.

    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/prompt.py#L321
    Modified so it uses numpy instead of torch.

    @param bboxes: The bounding boxes of the detected objects
    @type bboxes: np.ndarray
    @param masks: The masks of the detected objects
    @type masks: np.ndarray
    @param points: The points to adjust
    @type points: List[Tuple[int, int]]
    @param pointlabel: The point labels
    @type pointlabel: List[int]
    @param orig_shape: The original shape of the image
    @type orig_shape: Tuple[int, int] (height, width)
    @return: The modified masks
    @rtype: np.ndarray
    """
    if masks is not None:
        masks = format_results(bboxes, masks, 0)
        target_height, target_width = orig_shape
        h = masks[0]["segmentation"].shape[0]
        w = masks[0]["segmentation"].shape[1]
        if h != target_height or w != target_width:
            points = [
                [int(point[0] * w / target_width), int(point[1] * h / target_height)]
                for point in points
            ]
        onemask = np.zeros((h, w))
        for annotation in masks:
            mask = (
                annotation["segmentation"]
                if isinstance(annotation, dict)
                else annotation
            )
            for i, point in enumerate(points):
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                    onemask += mask
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                    onemask -= mask
        onemask = onemask >= 1
        masks = np.array([onemask])
    return masks


def adjust_bboxes_to_image_border(
    boxes: np.ndarray, image_shape: Tuple[int, int], threshold: int = 20
) -> np.ndarray:
    """
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/utils.py#L6 (Ultralytics)
    Adjust bounding boxes to stick to image border if they are within a certain threshold.

    @param boxes: Bounding boxes
    @type boxes: np.ndarray
    @param image_shape: Image shape
    @type image_shape: Tuple[int, int]
    @param threshold: Pixel threshold
    @type threshold: int
    @return: Adjusted bounding boxes
    @rtype: np.ndarray
    """
    # Image dimensions
    h, w = image_shape

    # Adjust boxes
    boxes[boxes[:, 0] < threshold, 0] = 0  # x1
    boxes[boxes[:, 1] < threshold, 1] = 0  # y1
    boxes[boxes[:, 2] > w - threshold, 2] = w  # x2
    boxes[boxes[:, 3] > h - threshold, 3] = h  # y2
    return boxes


def bbox_iou(
    box1: np.ndarray,
    boxes: np.ndarray,
    iou_thres: float = 0.9,
    image_shape: Tuple[int, int] = (640, 640),
    raw_output: bool = False,
) -> np.ndarray:
    """
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/utils.py#L30 (Ultralytics - rewritten to numpy)
    Compute the Intersection-Over-Union of a bounding box with respect to an array of other bounding boxes.

    @param box1: Array of shape (4, ) representing a single bounding box.
    @type box1: np.ndarray
    @param boxes: Array of shape (n, 4) representing multiple bounding boxes.
    @type boxes: np.ndarray
    @param iou_thres: IoU threshold
    @type iou_thres: float
    @param image_shape: Image shape (height, width)
    @type image_shape: Tuple[int, int]
    @param raw_output: If True, return the raw IoU values instead of the indices
    @type raw_output: bool
    @return: Indices of boxes with IoU > thres, or the raw IoU values if raw_output is True
    @rtype: np.ndarray
    """
    boxes = adjust_bboxes_to_image_border(boxes, image_shape)

    # Obtain coordinates for intersections
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    # Compute the area of intersection
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute the area of both individual boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Compute the area of union
    union = box1_area + box2_area - intersection

    # Compute the IoU
    iou = intersection / union  # Should be shape (n, )
    if raw_output:
        return iou if iou.size > 0 else np.array([])

    # Return indices of boxes with IoU > thres
    return np.flatnonzero(iou > iou_thres)


def decode_fastsam_output(
    outputs: List[np.ndarray],
    strides: List[int],
    anchors: List[Optional[np.ndarray]],
    img_shape: Tuple[int, int],
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    num_classes: int = 1,
) -> np.ndarray:
    """Decode the output of the FastSAM model.

    @param outputs: List of FastSAM outputs
    @type outputs: List[np.ndarray]
    @param strides: List of strides
    @type strides: List[int]
    @param anchors: List of anchors
    @type anchors: List[Optional[np.ndarray]]
    @param img_shape: Image shape
    @type img_shape: Tuple[int, int]
    @param conf_thres: Confidence threshold
    @type conf_thres: float
    @param iou_thres: IoU threshold
    @type iou_thres: float
    @param num_classes: Number of classes
    @type num_classes: int
    @return: NMS output
    @rtype: np.ndarray
    """
    output_nms = decode_yolo_output(
        yolo_outputs=outputs,
        strides=strides,
        anchors=anchors,
        kpts=None,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        num_classes=num_classes,
    )

    full_box = np.zeros(output_nms.shape[1])
    full_box[2], full_box[3], full_box[4], full_box[6:] = (
        img_shape[1],
        img_shape[0],
        1.0,
        1.0,
    )
    full_box = full_box.reshape((1, -1))
    critical_iou_index = bbox_iou(
        full_box[0][:4], output_nms[:, :4], iou_thres=0.9, image_shape=img_shape
    )
    if critical_iou_index.size > 0:
        idxs = critical_iou_index
        idx = idxs[np.argmax(output_nms[idxs, 4])]  # best confidence

        full_box[0, 4] = output_nms[idx, 4]
        full_box[0, 6:] = output_nms[idx, 6:]
        output_nms[idx] = full_box[0]

    return output_nms


def build_mask_coeffs(
    parsed_results: np.ndarray,
    masks_outputs_values: list[np.ndarray],
    protos_len: int,
) -> np.ndarray:
    """Gather mask coefficients for all detections, grouped by head.

    @param parsed_results: FastSAM decoded outputs
    @type parsed_results: np.ndarray
    @param masks_outputs_values: Model mask outputs
    @type masks_outputs_values: list[np.ndarray]
    @param protos_len: Number of protos
    @type protos_len: int
    """
    num_results = parsed_results.shape[0]
    seg_coeffs = parsed_results[:, 6:].astype(int)
    hi = seg_coeffs[:, 0]
    ai = seg_coeffs[:, 1]
    xi = seg_coeffs[:, 2]
    yi = seg_coeffs[:, 3]

    mask_coeffs = np.empty((num_results, protos_len), dtype=np.float32)
    coeff_indices = np.arange(protos_len)
    for head_idx, mask_values in enumerate(masks_outputs_values):
        selected = np.where(hi == head_idx)[0]
        if selected.size == 0:
            continue
        channel_indices = ai[selected, None] * protos_len + coeff_indices
        mask_coeffs[selected] = mask_values[0][
            channel_indices,
            yi[selected, None],
            xi[selected, None],
        ]
    return mask_coeffs


def process_masks(
    parsed_results: np.ndarray,
    mask_coeffs: np.ndarray,
    protos: np.ndarray,
    orig_shape: Tuple[int, int],
    mask_conf: float,
) -> np.ndarray:
    """Process output into full-size masks for all detections.

    @param parsed_results: FastSAM decoded outputs
    @type parsed_results: np.ndarray
    @param mask_coeffs: Mask coefficients
    @type mask_coeffs: np.ndarray
    @param protos: Protos from model output
    @type protos: np.ndarray
    @param orig_shape: Input shape of the model
    @type orig_shape: np.ndarray
    @param mask_conf: Mask confidence
    @type mask_conf: float
    """
    num_results = parsed_results.shape[0]
    out_w, out_h = orig_shape[0], orig_shape[1]

    results_masks = np.empty((num_results, out_h, out_w), dtype=np.uint8)
    bboxes_int = parsed_results[:, :4].astype(int)
    bboxes_clamped = np.clip(
        bboxes_int,
        a_min=np.array([0, 0, 0, 0], dtype=bboxes_int.dtype),
        a_max=np.array([out_w, out_h, out_w, out_h], dtype=bboxes_int.dtype),
    )
    mask_resized = np.empty((out_h, out_w), dtype=np.float32)

    for idx in range(num_results):
        mask_small = sigmoid(np.sum(protos * mask_coeffs[idx][:, None, None], axis=0))
        cv2.resize(
            mask_small,
            (out_w, out_h),
            dst=mask_resized,
            interpolation=cv2.INTER_NEAREST,
        )

        x1, y1, x2, y2 = bboxes_clamped[idx]

        if y1 > 0:
            mask_resized[:y1, :] = 0
        if y2 < out_h:
            mask_resized[y2:, :] = 0
        if x1 > 0:
            mask_resized[:, :x1] = 0
        if x2 < out_w:
            mask_resized[:, x2:] = 0

        np.greater(mask_resized, mask_conf, out=results_masks[idx])

    return results_masks


def merge_masks(masks: np.ndarray) -> np.ndarray:
    """Merge masks to a 2D array where each object is represented by a unique label.

    @param masks: 3D array of masks
    @type masks: np.ndarray
    @return: 2D array of masks
    @rtype: np.ndarray
    """
    if masks.ndim == 3:
        n, height, width = masks.shape
    else:
        raise ValueError("Masks must be a 3D array.")

    merged_masks = np.full((height, width), -1, dtype=np.int16)
    for i in range(n):
        merged_masks[masks[i] > 0] = i

    return merged_masks

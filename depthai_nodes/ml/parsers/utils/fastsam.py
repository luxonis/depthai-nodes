from typing import Tuple

import cv2
import numpy as np

from .yolo import non_max_suppression, parse_yolo_outputs, sigmoid


def box_prompt(masks, bbox, orig_shape):
    """
    Modifies the bounding box properties and calculates IoU between masks and bounding box.
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/prompt.py#L286
    Modified so it uses numpy instead of torch.
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


def format_results(bboxes, masks, filter=0):
    """
    Formats detection results into list of annotations each containing ID, segmentation, bounding box, score and
    area.
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/prompt.py#L56
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


def point_prompt(bboxes, masks, points, pointlabel, orig_shape):  # numpy
    """
    Adjusts points on detected masks based on user input and returns the modified results.
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/prompt.py#L321
    Modified so it uses numpy instead of torch.
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

    Args:
        boxes (np.ndarray): (n, 4)
        image_shape (tuple): (height, width)
        threshold (int): pixel threshold

    Returns:
        adjusted_boxes (np.ndarray): adjusted bounding boxes
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

    Args:
        box1 (np.ndarray): Array of shape (4, ) representing a single bounding box.
        boxes (np.ndarray): Array of shape (n, 4) representing multiple bounding boxes.
        iou_thres (float): IoU threshold.
        image_shape (tuple): (height, width).
        raw_output (bool): If True, return the raw IoU values instead of the indices.

    Returns:
        np.ndarray: Indices of boxes with IoU > thres, or the raw IoU values if raw_output is True.
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
    yolo_outputs,
    strides,
    anchors,
    img_shape: Tuple[int, int],
    conf_thres=0.5,
    iou_thres=0.45,
    num_classes=1,
):
    """
    Decode the bounding boxes

    Args:
        yolo_outputs (list): List of yolo outputs
        strides (list): List of strides
        anchors (list): List of anchors
        img_shape (tuple): Image shape (height, width)
        conf_thres (float): Confidence threshold
        iou_thres (float): IOU threshold

    Returns:
        output_nms (np.ndarray): NMS output
    """
    output = parse_yolo_outputs(yolo_outputs, strides, anchors, kpts=None)
    output_nms = non_max_suppression(
        output,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        num_classes=num_classes,
        kpts_mode=False,
    )[0]

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
        full_box[0][4] = output_nms[critical_iou_index][:, 4]
        full_box[0][6:] = output_nms[critical_iou_index][:, 6:]
        output_nms[critical_iou_index] = full_box

    return output_nms


def crop_mask(masks, box):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (numpy.ndarray): [h, w] array of masks
        boxes (numpy.ndarray): [4] array of bbox coordinates in (x1, y1, x2, y2) format

    Returns:
        numpy.ndarray: The masks are being cropped to the bounding box.
    """
    h, w = masks.shape
    x1, y1, x2, y2 = box
    r = np.arange(w).reshape(1, w)
    c = np.arange(h).reshape(h, 1)
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_single_mask(
    protos,
    mask_coeff,
    mask_conf,
    img_shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    mask = sigmoid(np.sum(protos * mask_coeff[..., np.newaxis, np.newaxis], axis=0))
    mask = cv2.resize(mask, img_shape, interpolation=cv2.INTER_NEAREST)
    mask = crop_mask(mask, np.array(bbox))
    return (mask > mask_conf).astype(np.uint8)

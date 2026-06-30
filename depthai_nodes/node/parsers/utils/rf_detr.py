import numpy as np

from depthai_nodes.node.parsers.utils import sigmoid, xyxy_to_xywh
from depthai_nodes.node.parsers.utils.bbox_format_converters import xywh_to_xyxy
from depthai_nodes.node.parsers.utils.masks_utils import process_single_mask_rfdetr


def compute_rfdetr_detections(
    boxes_tensor: np.ndarray,
    logits_tensor: np.ndarray,
    *,
    conf_threshold: float,
    max_det: int,
    label_names: list[str] | None,
    mask_conf: float,
    input_shape: tuple[int, int] | None,
    masks_tensor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None, np.ndarray | None]:
    """Decode RF-DETR detections and optional masks."""
    prob = sigmoid(logits_tensor)

    scores = np.max(prob, axis=2).squeeze()
    labels = np.argmax(prob, axis=2).squeeze()

    sorted_idx = np.argsort(scores)[::-1]

    effective_max_det = max_det
    if masks_tensor is not None:
        max_segmentation_instances = 255
        num_valid_instances = int(
            np.count_nonzero(scores[sorted_idx][:max_det] > conf_threshold)
        )
        effective_max_det = min(effective_max_det, max_segmentation_instances)
        if num_valid_instances > max_segmentation_instances:
            num_valid_instances = max_segmentation_instances

    scores = scores[sorted_idx][:effective_max_det]
    labels = labels[sorted_idx][:effective_max_det]
    boxes_cxcywh = boxes_tensor.squeeze()[sorted_idx][:effective_max_det]

    masks = None
    if masks_tensor is not None:
        masks = masks_tensor.squeeze()[sorted_idx][:effective_max_det]

    boxes = np.clip(xywh_to_xyxy(boxes_cxcywh), 0, 1)

    confidence_mask = scores > conf_threshold
    scores = scores[confidence_mask]
    labels = labels[confidence_mask]
    boxes = boxes[confidence_mask]
    boxes_cxcywh = boxes_cxcywh[confidence_mask]
    if masks is not None:
        masks = masks[confidence_mask]

    final_mask = None
    if masks is not None:
        if input_shape is None:
            raise ValueError(
                "RFDETRParser segmentation mode requires model input shape."
            )

        final_mask = np.full(input_shape, 255, dtype=np.uint8)
        for i, (mask_logits, bbox) in enumerate(zip(masks, boxes_cxcywh)):
            resized_mask = process_single_mask_rfdetr(
                mask_logits=mask_logits,
                mask_conf=mask_conf,
                bbox=bbox,
                input_shape=input_shape,
            )
            foreground = resized_mask > 0
            final_mask[(final_mask == 255) & foreground] = i

    boxes = xyxy_to_xywh(boxes)

    label_names_list = None
    if label_names:
        label_names_list = [
            (
                label_names[int(label)]
                if int(label) < len(label_names)
                else f"class_{int(label)}"
            )
            for label in labels
        ]

    return boxes, scores, labels.astype(int), label_names_list, final_mask

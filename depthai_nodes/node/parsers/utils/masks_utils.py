import cv2
import depthai as dai
import numpy as np


def probability_to_logit_threshold(probability: float) -> float:
    """Convert a probability threshold into the equivalent logit threshold."""
    if probability <= 0.0:
        return float("-inf")
    if probability >= 1.0:
        return float("inf")
    return float(np.log(probability / (1.0 - probability)))


def crop_mask(mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """It takes a mask and a bounding box, and returns a mask that is cropped to the
    bounding box.

    @param mask: [h, w] numpy array of a single mask
    @type mask: np.ndarray
    @param bbox: A numpy array of bbox coordinates in (x_center, y_center, width,
        height) format
    @type bbox: np.ndarray
    @return: A mask that is cropped to the bounding box
    @rtype: np.ndarray
    """
    h, w = mask.shape
    c_x, c_y, width, height = bbox
    x1 = c_x - width / 2
    y1 = c_y - height / 2
    x2 = c_x + width / 2
    y2 = c_y + height / 2
    r = np.arange(w).reshape(1, w)
    c = np.arange(h).reshape(h, 1)

    return mask * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_single_mask(
    protos: np.ndarray,
    mask_coeff: np.ndarray,
    mask_conf: float,
    bbox: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Process a single mask.

    @param protos: Protos.
    @type protos: np.ndarray
    @param mask_coeff: Mask coefficient.
    @type mask_coeff: np.ndarray
    @param mask_conf: Mask confidence.
    @type mask_conf: float
    @param bbox: A numpy array of bbox coordinates in (x_center, y_center, width,
        height) normalized format.
    @type bbox: np.ndarray
    @param output_shape: Target mask shape as (height, width).
    @type output_shape: tuple[int, int]
    @return: Processed binary mask resized to `output_shape`.
    @rtype: np.ndarray
    """
    _, mask_h, mask_w = protos.shape  # CHW
    scaled_bbox = bbox * np.array([mask_w, mask_h, mask_w, mask_h])

    mask_logits = np.sum(protos * mask_coeff[..., np.newaxis, np.newaxis], axis=0)
    mask_logits = crop_mask(mask_logits, scaled_bbox)
    mask_logits = cv2.resize(
        mask_logits,
        (output_shape[1], output_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    logit_threshold = probability_to_logit_threshold(mask_conf)
    return (mask_logits > logit_threshold).astype(np.uint8)


def get_segmentation_outputs(
    output: dai.NNData,
    mask_output_layer_names: list[str] | None = None,
    protos_output_layer_name: str | None = None,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Get the segmentation outputs from the Neural Network data."""
    # Get all the layer names
    layer_names = mask_output_layer_names or output.getAllLayerNames()
    mask_outputs = sorted([name for name in layer_names if "mask" in name])
    masks_outputs_values = [
        output.getTensor(
            o, dequantize=True, storageOrder=dai.TensorInfo.StorageOrder.NCHW
        ).astype(np.float32)
        for o in mask_outputs
    ]
    protos_output = output.getTensor(
        protos_output_layer_name or "protos_output",
        dequantize=True,
        storageOrder=dai.TensorInfo.StorageOrder.NCHW,
    ).astype(np.float32)
    protos_len = protos_output.shape[1]
    return masks_outputs_values, protos_output, protos_len


def process_single_mask_rfdetr(
    mask_logits: np.ndarray,
    mask_conf: float,
    bbox: np.ndarray,
    input_shape: tuple[int, int],
) -> np.ndarray:
    """Process a single RF-DETR instance segmentation mask.

    @param mask_logits: Mask logits for a single detection.
    @type mask_logits: np.ndarray
    @param mask_conf: Mask confidence threshold.
    @type mask_conf: float
    @param bbox: A numpy array of bbox coordinates in (x_center, y_center, width,
        height) normalized format.
    @type bbox: np.ndarray
    @param input_shape: Target output mask shape as (height, width).
    @type input_shape: tuple[int, int]
    @return: Processed mask resized to the model input shape.
    @rtype: np.ndarray
    """
    if mask_logits.ndim != 2:
        raise ValueError(
            f"RF-DETR mask logits should have shape (H, W), got {mask_logits.shape}."
        )

    resized_mask_logits = cv2.resize(
        mask_logits,
        (input_shape[1], input_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    logit_threshold = probability_to_logit_threshold(mask_conf)
    mask = (resized_mask_logits > logit_threshold).astype(np.uint8)

    scaled_bbox = bbox * np.array(
        [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
    )
    return crop_mask(mask, scaled_bbox)

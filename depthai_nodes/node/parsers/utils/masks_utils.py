import cv2
import depthai as dai
import numpy as np

from depthai_nodes.node.parsers.utils import sigmoid


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
    @return: Processed mask.
    @rtype: np.ndarray
    """
    c, mh, mw = protos.shape  # CHW
    scaled_bbox = bbox * np.array([mw, mh, mw, mh])
    mask = sigmoid(np.sum(protos * mask_coeff[..., np.newaxis, np.newaxis], axis=0))
    mask = crop_mask(mask, scaled_bbox)
    return (mask > mask_conf).astype(np.uint8)


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

    mask_h, mask_w = mask_logits.shape
    scaled_bbox = bbox * np.array([mask_w, mask_h, mask_w, mask_h])

    mask = sigmoid(mask_logits)
    mask = crop_mask(mask, scaled_bbox)
    mask = (mask > mask_conf).astype(np.uint8)

    return cv2.resize(
        mask,
        (input_shape[1], input_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

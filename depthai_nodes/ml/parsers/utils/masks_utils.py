import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function.

    @param x: Input tensor.
    @type x: np.ndarray
    @return: A result tensor after applying a sigmoid function on the given input.
    @rtype: np.ndarray
    """
    return 1 / (1 + np.exp(-x))


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

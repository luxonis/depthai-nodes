from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def local_maximum_filter(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a local maximum filter to the input array.

    @param x: Input array.
    @type x: np.ndarray
    @param kernel_size: Size of the local maximum filter.
    @type kernel_size: int
    @return: Output array after applying the local maximum filter.
    @rtype: np.ndarray
    """
    # Ensure input is a 4D array (e.g., batch, channels, height, width)
    if len(x.shape) != 4:
        raise ValueError("Input array must be 4-dimensional.")

    if x.shape[0] != 1 and x.shape[1] != 1:
        raise ValueError("Batch size and number of channels must be 1.")

    _, _, height, width = x.shape

    # Pad the input array
    pad_width = kernel_size // 2
    padded_x = np.pad(
        x,
        ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)),
        mode="constant",
    )

    # Use stride tricks to generate a view of the array with sliding windows
    shape = (
        padded_x.shape[0],
        padded_x.shape[1],
        height,
        width,
        kernel_size,
        kernel_size,
    )
    strides = (
        padded_x.strides[0],
        padded_x.strides[1],
        padded_x.strides[2],
        padded_x.strides[3],
        padded_x.strides[2],
        padded_x.strides[3],
    )

    sliding_window_view = np.lib.stride_tricks.as_strided(
        padded_x, shape=shape, strides=strides
    )

    # Compute the local maximum over the sliding windows
    local_max = np.max(sliding_window_view, axis=(4, 5))

    return local_max


def bilinear_grid_sample(
    im: np.ndarray, grid: np.ndarray, align_corners: bool = False
) -> np.ndarray:
    """Bilinear grid sample.

    @param im: Input image tensor.
    @type im: np.ndarray
    @param grid: Grid tensor.
    @type grid: np.ndarray
    @param align_corners: Whether to align corners.
    @type align_corners: bool
    @return: Output image tensor after applying bilinear grid sample.
    @rtype: np.ndarray
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.reshape(n, -1)
    y = y.reshape(n, -1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).reshape(n, 1, -1)
    wb = ((x1 - x) * (y - y0)).reshape(n, 1, -1)
    wc = ((x - x0) * (y1 - y)).reshape(n, 1, -1)
    wd = ((x - x0) * (y - y0)).reshape(n, 1, -1)

    # Apply padding
    im_padded = np.pad(
        im, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0
    )
    padded_h = h + 2
    padded_w = w + 2
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = np.clip(x0, 0, padded_w - 1)
    x1 = np.clip(x1, 0, padded_w - 1)
    y0 = np.clip(y0, 0, padded_h - 1)
    y1 = np.clip(y1, 0, padded_h - 1)

    im_padded = im_padded.reshape(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).reshape(n, 1, -1)
    x0_y1 = (x0 + y1 * padded_w).reshape(n, 1, -1)
    x1_y0 = (x1 + y0 * padded_w).reshape(n, 1, -1)
    x1_y1 = (x1 + y1 * padded_w).reshape(n, 1, -1)

    def gather(im_padded, idx):
        idx = idx.astype(np.int32)
        gathered = np.take_along_axis(im_padded, idx, axis=2)
        return gathered

    Ia = gather(im_padded, x0_y0)
    Ib = gather(im_padded, x0_y1)
    Ic = gather(im_padded, x1_y0)
    Id = gather(im_padded, x1_y1)

    result = (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)
    return result


def _get_kpts_heatmap(
    kpts: np.ndarray,
    softmax_temp: float = 1.0,
) -> np.ndarray:
    """Get the keypoints heatmap.

    @param kpts: Keypoints.
    @type kpts: np.ndarray
    @param softmax_temp: Softmax temperature.
    @type softmax_temp: float
    @return: Keypoints heatmap.
    @rtype: np.ndarray
    """
    kpts = np.exp(kpts * softmax_temp)
    scores = kpts / np.sum(kpts, axis=1, keepdims=True)
    scores = scores[:, :64]
    B, _, H, W = scores.shape
    heatmap = np.transpose(scores, (0, 2, 3, 1)).reshape(B, H, W, 8, 8)
    heatmap = np.transpose(heatmap, (0, 1, 3, 2, 4)).reshape(B, 1, H * 8, W * 8)
    return heatmap


def _nms(
    x: np.ndarray,
    threshold: float = 0.05,
    kernel_size: int = 5,
) -> np.ndarray:
    """Non-Maximum Suppression.

    @param x: Input array.
    @type x: np.ndarray
    @param threshold: Non-maximum suppression threshold.
    @type threshold: float
    @param kernel_size: Size of the local maximum filter.
    @type kernel_size: int
    @return: Output array after applying non-maximum suppression.
    @rtype: np.ndarray
    """
    # Non-Maximum Suppression
    B, _, H, W = x.shape
    local_max = local_maximum_filter(x, kernel_size)
    pos = (x == local_max) & (x > threshold)

    pos_batched = [np.fliplr(np.argwhere(k)[:, 1:]) for k in pos]

    pad_val = max(len(k) for k in pos_batched)
    pos_array = np.zeros((B, pad_val, 2), dtype=int)

    for b, kpts in enumerate(pos_batched):
        pos_array[b, : len(kpts)] = kpts

    return pos_array


def detect_and_compute(
    feats: np.ndarray,
    kpts: np.ndarray,
    resize_rate_w: float,
    resize_rate_h: float,
    input_size: Tuple[int, int],
    top_k: int = 4096,
) -> List[Dict[str, Any]]:
    """Detect and compute keypoints.

    @param feats: Features.
    @type feats: np.ndarray
    @param kpts: Keypoints.
    @type kpts: np.ndarray
    @param resize_rate_w: Resize rate for width.
    @type resize_rate_w: float
    @param resize_rate_h: Resize rate for height.
    @type resize_rate_h: float
    @param input_size: Input size.
    @type input_size: Tuple[int, int]
    @param top_k: Maximum number of keypoints to keep.
    @type top_k: int
    @return: List of dictionaries containing keypoints, scores, and descriptors.
    @rtype: List[Dict[str, Any]]
    """
    norm = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / norm

    kpts_heats = _get_kpts_heatmap(kpts)
    mkpts = _nms(kpts_heats, threshold=0.05, kernel_size=5)  # int64

    # Numpy implementation of normgrid
    div_array = np.array([input_size[0] - 1, input_size[1] - 1], dtype=mkpts.dtype)
    grid = 2.0 * (mkpts / div_array) - 1.0
    grid = np.expand_dims(grid, axis=2)

    if grid.size == 0:
        return None

    # Numpy implementation of F.grid_sample
    map_x = grid[..., 0].reshape(-1).astype(np.float32)
    map_y = grid[..., 1].reshape(-1).astype(np.float32)
    remapped = cv2.remap(
        kpts_heats[0, 0],
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    nearest_result = np.expand_dims(remapped, axis=0)

    # Numpy implementation of F.grid_sample
    remapped = cv2.remap(
        kpts_heats[0, 0],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    bilinear_result = np.expand_dims(remapped, axis=0)
    scores = (nearest_result * bilinear_result).reshape(1, -1)

    scores = scores.astype(np.float32)
    mkpts = mkpts.astype(np.int64)

    scores[np.all(mkpts == 0, axis=-1)] = -1

    idxs = np.argsort(-scores)
    mkpts_x = np.take_along_axis(mkpts[..., 0], idxs, axis=-1)[:, :top_k]
    mkpts_y = np.take_along_axis(mkpts[..., 1], idxs, axis=-1)[:, :top_k]
    mkpts = np.stack([mkpts_x, mkpts_y], axis=-1)
    scores = np.take_along_axis(scores, idxs, axis=-1)[:, :top_k]

    div_array = np.array([input_size[0] - 1, input_size[1] - 1], dtype=mkpts.dtype)
    grid = 2.0 * (mkpts / div_array) - 1.0
    grid = np.expand_dims(grid, axis=2)
    map_x = grid[..., 0].reshape(-1).astype(np.float32)
    map_y = grid[..., 1].reshape(-1).astype(np.float32)
    mkpts = mkpts.astype(np.float32)

    feats = bilinear_grid_sample(feats, grid, align_corners=False)
    feats = feats.transpose(0, 2, 3, 1).squeeze(-2)

    norm = np.linalg.norm(feats, axis=-1, keepdims=True)
    feats = feats / norm

    mkpts = mkpts.astype(np.float32)
    mkpts *= np.array([resize_rate_w, resize_rate_h])[None, None, :]

    valid = scores > 0
    result = []
    valid = valid[0]
    result.append(
        {
            "keypoints": mkpts[0][valid],
            "scores": scores[0][valid],
            "descriptors": feats[0][valid],
        }
    )

    return result


def _match_mkpts(
    feats1: np.ndarray, feats2: np.ndarray, min_cossim: float = 0.62
) -> Tuple[np.ndarray, np.ndarray]:
    """Match features.

    @param feats1: Features 1.
    @type feats1: np.ndarray
    @param feats2: Features 2.
    @type feats2: np.ndarray
    @param min_cossim: Minimum cosine similarity.
    @type min_cossim: float
    @return: Matched features.
    @rtype: Tuple[np.ndarray, np.ndarray]
    """
    cossim = feats1 @ feats2.T
    cossim_t = feats2 @ feats1.T
    match12 = np.argmax(cossim, axis=1)
    match21 = np.argmax(cossim_t, axis=1)

    idx0 = np.arange(len(match12))
    mutual = match21[match12] == idx0

    if min_cossim > 0:
        max_cossim = np.max(cossim, axis=1)
        good = max_cossim > min_cossim
        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]
    else:
        idx0 = idx0[mutual]
        idx1 = match12[mutual]

    return idx0, idx1


def match(
    result1: Dict[str, Any], result2: Dict[str, Any], min_cossim: float = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """Match keypoints.

    @param result1: Result 1.
    @type result1: Dict[str, Any]
    @param result2: Result 2.
    @type result2: Dict[str, Any]
    @param min_cossim: Minimum cosine similarity.
    @type min_cossim: float
    @return: Matched keypoints.
    @rtype: Tuple[np.ndarray, np.ndarray]
    """
    indexes1, indexes2 = _match_mkpts(
        result1["descriptors"],
        result2["descriptors"],
        min_cossim=min_cossim,
    )

    mkpts0 = result1["keypoints"][indexes1]
    mkpts1 = result2["keypoints"][indexes2]

    return mkpts0, mkpts1

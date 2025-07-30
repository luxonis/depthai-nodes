from typing import List, Tuple

import numpy as np


def decode_scores_and_points(
    tpMap: np.ndarray, heat: np.ndarray, topk_n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the scores and points from the neural network output tensors. Used for
    MLSD model.

    @param tpMap: Tensor containing the vector map.
    @type tpMap: np.ndarray
    @param heat: Tensor containing the heat map.
    @type heat: np.ndarray
    @param topk_n: Number of top candidates to keep.
    @type topk_n: int
    @return: Detected points, confidence scores for the detected points, and vector map.
    @rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    _, _, h, w = tpMap.shape
    displacement = tpMap[0, 1:5]  # shape (4, h, w)

    # Flatten heatmap for fast topk
    heat_flat = heat.flatten()
    if topk_n > heat_flat.size:
        topk_n = heat_flat.size

    # Top-K indices (unsorted)
    indices_np = np.argpartition(heat_flat, -topk_n)[-topk_n:]
    # Optionally: sort true top-k in descending score order
    sorted_idx = indices_np[np.argsort(-heat_flat[indices_np])]
    pts_score = heat_flat[sorted_idx]

    # Convert flat indices to 2D (y, x)
    yy_np, xx_np = np.divmod(sorted_idx, w)
    pts = np.stack((yy_np, xx_np), axis=1)

    vmap = np.transpose(displacement, (1, 2, 0))  # (h, w, 4)

    return pts, pts_score, vmap


def get_lines(
    pts: np.ndarray,
    pts_score: np.ndarray,
    vmap: np.ndarray,
    score_thr: float,
    dist_thr: float,
    input_size: int = 512,
) -> Tuple[np.ndarray, List[float]]:
    """Get lines from the detected points and scores. The lines are filtered by the
    score threshold and distance threshold. Used for MLSD model.

    @param pts: Detected points.
    @type pts: np.ndarray
    @param pts_score: Confidence scores for the detected points.
    @type pts_score: np.ndarray
    @param vmap: Vector map.
    @type vmap: np.ndarray
    @param score_thr: Confidence score threshold for detected lines.
    @type score_thr: float
    @param dist_thr: Distance threshold for merging lines.
    @type dist_thr: float
    @param input_size: Input size of the model.
    @type input_size: int
    @return: Detected lines and their confidence scores.
    @rtype: Tuple[np.ndarray, List[float]]
    """
    # Extract coordinates for all points
    ys, xs = pts[:, 0], pts[:, 1]
    # Vectorized gather
    disp = vmap[ys, xs, :]  # shape: (num_pts, 4)
    start_xy = np.stack([xs + disp[:, 0], ys + disp[:, 1]], axis=1)
    end_xy = np.stack([xs + disp[:, 2], ys + disp[:, 3]], axis=1)

    # Compute line length (distance)
    dists = np.linalg.norm(start_xy - end_xy, axis=1)

    # Apply both thresholds in one go
    keep = (pts_score > score_thr) & (dists > dist_thr)

    # Stack lines and normalize to [0,1] for input_size
    lines = np.hstack([start_xy[keep], end_xy[keep]]).astype(np.float32)
    lines = 2 * lines / input_size  # scale: 256→512

    return lines, pts_score[keep].tolist()

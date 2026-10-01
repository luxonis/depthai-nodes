import numpy as np


def decode_scores_and_points(
    tpMap: np.ndarray, heat: np.ndarray, topk_n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the scores and points from the neural network output tensors. Used for
    MLSD model.

    Args:
        tpMap: Tensor containing the vector map.
        heat: Tensor containing the heat map.
        topk_n: Number of top candidates to keep.

    Returns:
        Detected points, confidence scores for the detected points, and vector map.
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
) -> tuple[np.ndarray, list[float]]:
    """Get lines from the detected points and scores. The lines are filtered by the
    score threshold and distance threshold. Used for MLSD model.

    Args:
        pts: Detected points.
        pts_score: Confidence scores for the detected points.
        vmap: Vector map.
        score_thr: Confidence score threshold for detected lines.
        dist_thr: Minimum line length in output-map pixels.
        input_size: Input size of the model.

    Returns:
        Detected lines and their confidence scores.
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


def compute_mlsd_lines(
    tpMap: np.ndarray,
    heat: np.ndarray,
    *,
    topk_n: int,
    score_thr: float,
    dist_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode line segments from M-LSD displacement and heat tensors.

    Args:
        tpMap: Four-dimensional line-displacement tensor in NCHW layout.
        heat: Heat tensor used to rank line-center candidates.
        topk_n: Maximum number of line-center candidates to examine.
        score_thr: Minimum candidate score.
        dist_thr: Minimum line length in output-map pixels.

    Returns:
        Normalized endpoint coordinates of shape ``(N, 4)`` and float32 line scores.

    Raises:
        ValueError: If ``tpMap`` is not four-dimensional.
    """
    if len(tpMap.shape) != 4:
        raise ValueError("Invalid shape of the tpMap tensor. Should be 4D.")

    pts, pts_score, vmap = decode_scores_and_points(tpMap, heat, topk_n)
    lines, scores = get_lines(pts, pts_score, vmap, score_thr, dist_thr)
    return lines, np.array(scores, dtype=np.float32)

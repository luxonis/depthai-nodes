import numpy as np


def compute_hrnet_keypoints(
    heatmaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized keypoints and scores from HRNet heatmaps."""
    maps = np.asarray(heatmaps)

    if maps.shape[0] == 1:
        maps = maps[0]

    if maps.ndim != 3:
        raise ValueError(f"Expected 3D output tensor, got {maps.ndim}D.")

    _, map_h, map_w = maps.shape

    scores = np.array([np.max(heatmap) for heatmap in maps], dtype=np.float32)
    scores = np.clip(scores, 0, 1)

    keypoints = np.array(
        [np.unravel_index(heatmap.argmax(), heatmap.shape) for heatmap in maps],
        dtype=np.float32,
    )
    keypoints = keypoints[:, ::-1] / np.array([map_w, map_h], dtype=np.float32)

    return keypoints, scores

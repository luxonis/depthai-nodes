from typing import List, Tuple

import numpy as np


def decode_scores_and_points(
    tpMap: np.ndarray, heat: np.ndarray, topk_n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, c, h, w = tpMap.shape
    displacement = tpMap[:, 1:5, :, :][0]

    indices_np = np.argpartition(heat, -topk_n)[-topk_n:]
    pts_score = heat[indices_np]
    yy_np = np.floor_divide(indices_np, w).reshape(-1, 1)
    xx_np = np.fmod(indices_np, w).reshape(-1, 1)
    pts = np.hstack((yy_np, xx_np))

    vmap = displacement.transpose((1, 2, 0))
    return pts, pts_score, vmap


def get_lines(
    pts: np.ndarray,
    pts_score: np.ndarray,
    vmap: np.ndarray,
    score_thr: float,
    dist_thr: float,
    input_size: int = 512,
) -> Tuple[np.ndarray, List[float]]:
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    line_scores = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
            line_scores.append(score)

    lines = 2 * np.array(segments_list)  # 256 > 512
    lines /= input_size
    return lines, line_scores

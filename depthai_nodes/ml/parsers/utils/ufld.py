from typing import List, Tuple

import numpy as np


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=0)


def decode_ufld(
    anchors: List[int],
    griding_num: int,
    cls_num_per_lane: int,
    INPUT_WIDTH: int,
    INPUT_HEIGHT: int,
    y: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    col_sample = np.linspace(0, INPUT_WIDTH - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    img_w, img_h = INPUT_WIDTH, INPUT_HEIGHT

    out_j = y
    out_j = out_j[:, ::-1, :]
    prob = softmax(out_j[:-1, :, :])

    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    points = [[] for _ in range(out_j.shape[1])]

    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (
                        int(out_j[k, i] * col_sample_w * img_w / INPUT_WIDTH) - 1,
                        int(img_h * (anchors[cls_num_per_lane - 1 - k] / INPUT_HEIGHT))
                        - 1,
                    )
                    points[i].append(ppp)

    return points

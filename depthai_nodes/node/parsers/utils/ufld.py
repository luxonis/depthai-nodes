from typing import List, Tuple

import numpy as np

from depthai_nodes.node.parsers.utils import softmax


def decode_ufld(
    anchors: List[int],
    griding_num: int,
    cls_num_per_lane: int,
    input_width: int,
    input_height: int,
    y: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    col_sample = np.linspace(0, input_width - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = y
    out_j = out_j[:, ::-1, :]
    prob = softmax(out_j[:-1, :, :], axis=0)

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
                    ppp = [
                        int(out_j[k, i] * col_sample_w) - 1,
                        int(
                            input_height
                            * (anchors[cls_num_per_lane - 1 - k] / input_height)
                        )
                        - 1,
                    ]
                    # Normalize points to [0,1] range
                    ppp[0] /= input_width
                    ppp[1] /= input_height
                    ppp = tuple(ppp)

                    points[i].append(ppp)

    return points

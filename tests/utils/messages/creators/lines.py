from typing import List

import numpy as np

import depthai_nodes.message.creators as creators

from .constants import COLLECTIONS, SCORES


def create_lines(
    lines: np.ndarray = COLLECTIONS["lines"],
    scores: List[float] = SCORES,
):
    return creators.create_line_detection_message(
        lines=np.array(lines), scores=np.array(scores)
    )

import numpy as np

import depthai_nodes.message.creators as creators

from .constants import DETECTIONS


def create_keypoints(
    keypoints: np.ndarray = DETECTIONS["keypoints"][0],
    scores: np.ndarray = DETECTIONS["keypoints_scores"][0],
):
    return creators.create_keypoints_message(keypoints=keypoints, scores=scores)

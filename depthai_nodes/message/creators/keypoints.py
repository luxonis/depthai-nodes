from typing import Iterable, Optional

import depthai as dai
import numpy as np


def create_keypoints_message(
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    confidence_threshold: Optional[float] = None,
    edges: Optional[Iterable[tuple[int, int]]] = None,
    label_names: Optional[list[str]] = None,
) -> dai.KeypointsList:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] not in (2, 3):
        raise ValueError(
            f"Keypoints must have shape (N, 2) or (N, 3), got {keypoints.shape}."
        )

    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 1 or scores.shape[0] != keypoints.shape[0]:
            raise ValueError(
                "scores must be a 1D array with one confidence value per keypoint."
            )

    if confidence_threshold is None:
        confidence_threshold = 0.0

    msg = dai.KeypointsList()
    keypoints_list: list[dai.Keypoint] = []
    for idx, point in enumerate(keypoints):
        confidence = 1.0 if scores is None else float(scores[idx])
        if confidence < confidence_threshold:
            continue

        keypoint = dai.Keypoint()
        keypoint.imageCoordinates.x = float(point[0])
        keypoint.imageCoordinates.y = float(point[1])
        if point.shape[0] == 3:
            keypoint.imageCoordinates.z = float(point[2])
        keypoint.confidence = confidence
        keypoint.label = idx
        if label_names is not None and idx < len(label_names):
            keypoint.labelName = label_names[idx]
        keypoints_list.append(keypoint)

    msg.setKeypoints(keypoints_list)
    if edges is not None:
        msg.setEdges(list(edges))
    return msg

from typing import List, Optional, Tuple, Union

import depthai as dai
import numpy as np


def create_keypoints_message(
    keypoints: Union[np.ndarray, List[List[float]]],
    scores: Union[np.ndarray, List[float], None] = None,
    confidence_threshold: Optional[float] = None,
    label_names: Optional[List[str]] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
) -> dai.KeypointsList:
    """Create a native DepthAI keypoints message."""

    if not isinstance(keypoints, (np.ndarray, list)):
        raise ValueError(
            f"Keypoints should be numpy array or list, got {type(keypoints)}."
        )

    if scores is not None:
        if not isinstance(scores, (np.ndarray, list)):
            raise ValueError(
                f"Scores should be numpy array or list, got {type(scores)}."
            )

        if len(keypoints) != len(scores):
            raise ValueError(
                "Keypoints and scores should have the same length. Got {} keypoints and {} scores.".format(
                    len(keypoints), len(scores)
                )
            )

        if not all(isinstance(score, (float, np.floating)) for score in scores):
            raise ValueError("Scores should only contain float values.")
        if not all(0 <= score <= 1 for score in scores):
            raise ValueError("Scores should only contain values between 0 and 1.")

    if confidence_threshold is not None:
        if not isinstance(confidence_threshold, float):
            raise ValueError(
                f"The confidence_threshold should be float, got {type(confidence_threshold)}."
            )

        if not (0 <= confidence_threshold <= 1):
            raise ValueError(
                f"The confidence_threshold should be between 0 and 1, got confidence_threshold {confidence_threshold}."
            )

    dimension = 0
    if len(keypoints) != 0:
        dimension = len(keypoints[0])
        if dimension != 2 and dimension != 3:
            raise ValueError(
                f"All keypoints should be of dimension 2 or 3, got dimension {dimension}."
            )

    if isinstance(keypoints, list):
        for keypoint in keypoints:
            if not isinstance(keypoint, list):
                raise ValueError(
                    f"Keypoints should be list of lists or np.array, got list of {type(keypoint)}."
                )
            if len(keypoint) != dimension:
                raise ValueError(
                    "All keypoints have to be of same dimension e.g. [x, y] or [x, y, z], got mixed inner dimensions."
                )
            for coord in keypoint:
                if not isinstance(coord, (float, np.floating)):
                    raise ValueError(
                        f"Keypoints inner list should contain only float, got {type(coord)}."
                    )

    if label_names is not None:
        if not isinstance(label_names, list):
            raise ValueError(f"label_names should be list, got {type(label_names)}.")
        if not all(isinstance(label, str) for label in label_names):
            raise ValueError("label_names should be a list of strings.")

    if edges is not None:
        if not isinstance(edges, list):
            raise ValueError(f"Edges should be list, got {type(edges)}.")
        if not all(
            isinstance(edge, tuple)
            and len(edge) == 2
            and all(isinstance(i, int) for i in edge)
            for edge in edges
        ):
            raise ValueError("Edges should be a list of tuples of integers.")

    keypoints = np.array(keypoints, dtype=float)
    if scores is not None:
        scores = np.array(scores)

    use_3d = False
    if len(keypoints) != 0:
        if len(keypoints.shape) != 2:
            raise ValueError(
                f"Keypoints should be of shape (N,2 or 3) got {keypoints.shape}."
            )
        if keypoints.shape[1] not in (2, 3):
            raise ValueError(
                f"Keypoints should be of shape (N,2 or 3) got {keypoints.shape}."
            )
        use_3d = keypoints.shape[1] == 3

    keypoints_msg = dai.KeypointsList()
    points = []
    included_keypoints = []

    for i, keypoint in enumerate(keypoints):
        if scores is not None and confidence_threshold is not None and scores[i] < confidence_threshold:
            continue

        pt = dai.Keypoint()
        pt.imageCoordinates = dai.Point3f(
            float(keypoint[0]),
            float(keypoint[1]),
            float(keypoint[2]) if use_3d else 0.0,
        )
        pt.confidence = float(scores[i]) if scores is not None else -1.0
        if label_names is not None:
            pt.labelName = label_names[i]
        points.append(pt)
        included_keypoints.append(i)

    keypoints_msg.setKeypoints(points)

    if edges is not None:
        filtered_edges = []
        index_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(included_keypoints)
        }

        for edge in edges:
            if edge[0] in included_keypoints and edge[1] in included_keypoints:
                filtered_edges.append(
                    (index_mapping[edge[0]], index_mapping[edge[1]])
                )

        keypoints_msg.setEdges(filtered_edges)

    return keypoints_msg

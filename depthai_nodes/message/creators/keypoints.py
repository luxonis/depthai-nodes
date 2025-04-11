from typing import List, Optional, Tuple, Union

import numpy as np

from depthai_nodes import Keypoint, Keypoints


def create_keypoints_message(
    keypoints: Union[np.ndarray, List[List[float]]],
    scores: Union[np.ndarray, List[float], None] = None,
    confidence_threshold: Optional[float] = None,
    label_names: Optional[List[str]] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
) -> Keypoints:
    """Create a DepthAI message for the keypoints.

    NOTE: If you provide a confidence threshold for filtering the keypoints based on scores, the edges will be filtered to only include the edges between the keypoints that are present in the filtered keypoints. This is done to ensure that the edges are only drawn between the keypoints that are present in the filtered keypoints. You can always access the full set of edges in the model's NN archive.

    @param keypoints: Detected 2D or 3D keypoints of shape (N,2 or 3) meaning [...,[x, y],...] or [...,[x, y, z],...].
    @type keypoints: np.ndarray or List[List[float]]
    @param scores: Confidence scores of the detected keypoints. Defaults to None.
    @type scores: Union[np.ndarray, List[float], None]
    @param confidence_threshold: Confidence threshold of keypoint detections. Defaults to None.
    @type confidence_threshold: Optional[float]
    @param label_names: label_names of the detected keypoints. Defaults to None.
    @type label_names: Optional[List[str]]
    @param edges: Connection pairs of the detected keypoints. Defaults to None. Example: [[0,1], [1,2], [2,3], [3,0]] shows that keypoint 0 is connected to keypoint 1, keypoint 1 is connected to keypoint 2, etc.
    @type edges: Optional[List[Tuple[int, int]]]
    @return: Keypoints message containing the detected keypoints.
    @rtype: Keypoints

    @raise ValueError: If the keypoints are not a numpy array or list.
    @raise ValueError: If the scores are not a numpy array or list.
    @raise ValueError: If scores and keypoints do not have the same length.
    @raise ValueError: If score values are not floats.
    @raise ValueError: If score values are not between 0 and 1.
    @raise ValueError: If the confidence threshold is not a float.
    @raise ValueError: If the confidence threshold is not between 0 and 1.
    @raise ValueError: If the keypoints are not of shape (N,2 or 3).
    @raise ValueError: If the keypoints 2nd dimension is not of size E{2} or E{3}.
    """

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

    keypoints = np.array(keypoints)
    if scores is not None:
        scores = np.array(scores)

    if len(keypoints) != 0:
        if len(keypoints.shape) != 2:
            raise ValueError(
                f"Keypoints should be of shape (N,2 or 3) got {keypoints.shape}."
            )

        use_3d = keypoints.shape[1] == 3

    keypoints_msg = Keypoints()
    points = []
    included_keypoints = []

    for i, keypoint in enumerate(keypoints):
        if scores is not None and confidence_threshold is not None:
            if scores[i] < confidence_threshold:
                continue
        pt = Keypoint()
        pt.x = float(keypoint[0])
        pt.y = float(keypoint[1])
        pt.z = float(keypoint[2]) if use_3d else 0.0
        if scores is not None:
            pt.confidence = float(scores[i])
        if label_names is not None:
            pt.label_name = label_names[i]
        points.append(pt)
        included_keypoints.append(i)

    keypoints_msg.keypoints = points

    if edges is not None:
        filtered_edges = []
        # Create a mapping from original indices to new indices
        index_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(included_keypoints)
        }

        for edge in edges:
            # Only include edges where both endpoints exist in the filtered keypoints
            if edge[0] in included_keypoints and edge[1] in included_keypoints:
                # Map the old indices to the new indices
                new_edge = [index_mapping[edge[0]], index_mapping[edge[1]]]
                filtered_edges.append(new_edge)

        keypoints_msg.edges = filtered_edges
    return keypoints_msg

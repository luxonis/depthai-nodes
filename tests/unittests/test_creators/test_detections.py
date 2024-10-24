import re

import numpy as np
import pytest

from depthai_nodes.ml.messages import (
    ImgDetectionsExtended,
)
from depthai_nodes.ml.messages.creators.detection import create_detection_message


def test_bboxes_not_numpy_array():
    with pytest.raises(
        ValueError, match="Bounding boxes should be a numpy array, got <class 'int'>."
    ):
        create_detection_message(bboxes=1, scores=1)


def test_bbox_dim():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Bounding boxes should be a 2D array like [... , [x_center, y_center, width, height], ... ], got (1,)."
        ),
    ):
        create_detection_message(bboxes=np.array([1.0]), scores=1)


def test_bbox_dim_4():
    with pytest.raises(
        ValueError,
        match=re.escape("Bounding boxes should be of shape (n, 4), got (1, 3)."),
    ):
        create_detection_message(bboxes=np.array([[0.25, 0.25, 0.25]]), scores=1)


def test_scores_not_numpy_array():
    with pytest.raises(
        ValueError, match="Scores should be a numpy array, got <class 'int'>."
    ):
        create_detection_message(bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]), scores=1)


def test_scores_shape():
    with pytest.raises(
        ValueError,
        match=re.escape("Scores should be of shape (N,), got (1, 1)."),
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]), scores=np.array([[0.1]])
        )


def test_scores_length():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Scores should have same length as bboxes, got 2 scores and 1 bounding boxes."
        ),
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]), scores=np.array([0.1, 0.2])
        )


def test_labels_list():
    with pytest.raises(
        ValueError, match="Labels should be a numpy array, got <class 'int'>."
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]), scores=np.array([0.1]), labels=1
        )


def test_labels_bbox_lengths():
    with pytest.raises(
        ValueError,
        match="Labels should have same length as bboxes, got 2 labels and 1 bounding boxes.",
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            labels=np.array([0.1, 0.2]),
        )


def test_angles_list():
    with pytest.raises(
        ValueError, match="Angles should be a numpy array, got <class 'int'>."
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]), scores=np.array([0.1]), angles=1
        )


def test_angles_bbox_lengths():
    with pytest.raises(
        ValueError,
        match="Angles should have same length as bboxes, got 2 angles and 1 bounding boxes.",
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            angles=np.array([0.1, 0.2]),
        )


def test_angles_values():
    with pytest.raises(
        ValueError, match="Angles should be between -360 and 360, got 999."
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            angles=np.array([999]),
        )


def test_keypoints_list():
    with pytest.raises(
        ValueError, match="Keypoints should be a numpy array, got <class 'int'>."
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]), scores=np.array([0.1]), keypoints=1
        )


def test_keypoints_shape():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Keypoints should be of shape (N, M, 2 or 3) meaning [..., [x, y] or [x, y, z], ...], got (1, 3)."
        ),
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            keypoints=np.array([[1.0, 2.0, 3.0]]),
        )


def test_keypoints_bbox_lengths():
    with pytest.raises(
        ValueError,
        match="Keypoints should have same length as bboxes, got 2 keypoints and 1 bounding boxes",
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            keypoints=np.array([[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]]),
        )


def test_keypoints_scores_without_keypoints():
    with pytest.raises(
        ValueError,
        match="Keypoints scores should be provided only if keypoints are provided.",
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            keypoints_scores=1,
        )


def test_keypoints_scores_list():
    with pytest.raises(
        ValueError, match="Keypoints scores should be a numpy array, got <class 'int'>."
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            keypoints=np.array([[[1.0, 2.0, 3.0]]]),
            keypoints_scores=1,
        )


def test_keypoints_scores_lengths():
    with pytest.raises(
        ValueError,
        match="Keypoints scores should have same length as keypoints, got 2 keypoints scores and 1 keypoints.",
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            keypoints=np.array([[[1.0, 2.0, 3.0]]]),
            keypoints_scores=np.array([[0.1], [0.2]]),
        )


def test_number_of_keypoints_scores_per_detection():
    with pytest.raises(
        ValueError,
        match="Number of keypoints scores per detection should be the same as number of keypoints per detection, got 2 keypoints scores and 1 keypoints.",
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            keypoints=np.array([[[1.0, 2.0, 3.0]]]),
            keypoints_scores=np.array([[0.1, 0.2]]),
        )


def test_keypoints_scores_values():
    with pytest.raises(ValueError, match="Keypoints scores should be between 0 and 1."):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            keypoints=np.array([[[1.0, 2.0, 3.0]]]),
            keypoints_scores=np.array([[-0.1]]),
        )


def test_masks_list():
    with pytest.raises(
        ValueError, match="Masks should be a numpy array, got <class 'int'>."
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]), scores=np.array([0.1]), masks=1
        )


def test_masks_shape():
    with pytest.raises(
        ValueError,
        match=re.escape("Masks should be of shape (H, W), got (3,)."),
    ):
        create_detection_message(
            bboxes=np.array([[0.25, 0.25, 0.25, 0.25]]),
            scores=np.array([0.1]),
            masks=np.array([1, 2, 3]),
        )


def test_only_bboxes_scores():
    bboxes = np.array(
        [[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2]]
    )
    scores = np.array([0.1, 0.2, 0.3])

    message = create_detection_message(bboxes=bboxes, scores=scores)

    assert isinstance(message, ImgDetectionsExtended)
    assert all(detection.label == -1 for detection in message.detections)
    for i, detection in enumerate(message.detections):
        assert detection.x_center == bboxes[i, 0]
        assert detection.y_center == bboxes[i, 1]
        assert detection.width == bboxes[i, 2]
        assert detection.height == bboxes[i, 3]
        assert np.isclose(detection.confidence, scores[i])


def test_bboxes_scores_labels():
    bboxes = np.array(
        [[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2]]
    )
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 2, 3])

    message = create_detection_message(bboxes=bboxes, scores=scores, labels=labels)

    for i, label in enumerate(labels):
        assert message.detections[i].label == label


def test_bboxes_scores_keypoints():
    bboxes = np.array(
        [[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2]]
    )
    scores = np.array([0.1, 0.2, 0.3])
    keypoints = np.array([[(0.0, 0.0), (0.1, 0.1)], [(0.2, 0.2), (0.3, 0.3)], [(0.4, 0.4), (0.5, 0.5)]])

    message = create_detection_message(
        bboxes=bboxes, scores=scores, keypoints=keypoints
    )

    assert isinstance(message, ImgDetectionsExtended)

    for i, detection in enumerate(message.detections):
        for ii, keypoint in enumerate(detection.keypoints):
            assert keypoint.x == keypoints[i][ii][0]
            assert keypoint.y == keypoints[i][ii][1]


if __name__ == "__main__":
    pytest.main()

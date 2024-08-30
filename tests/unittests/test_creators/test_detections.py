import re

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.ml.messages import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
)
from depthai_nodes.ml.messages.creators.detection import create_detection_message


def test_not_numpy_array():
    with pytest.raises(
        ValueError, match="Bounding boxes should be numpy array, got <class 'list'>."
    ):
        create_detection_message([1, 2, 3], [0.1, 0.2, 0.3], None, None)


def test_bbox_dim():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Bounding boxes should be of shape (N,4) meaning [...,[x_min, y_min, x_max, y_max],...], got (1,)."
        ),
    ):
        create_detection_message(np.array([1.0]), [0.1], None, None)


def test_bbox_dim_4():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Bounding boxes 2nd dimension should be of size 4 e.g. [x_min, y_min, x_max, y_max] got 3."
        ),
    ):
        create_detection_message(np.array([[1.0, 2.0, 3.0]]), [0.1], None, None)


def test_bbox_valid():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Bounding boxes should be in format [x_min, y_min, x_max, y_max] where xmin < xmax and ymin < ymax."
        ),
    ):
        create_detection_message(np.array([[3.0, 2.0, 1.0, 4.0]]), [0.1], None, None)


def test_scores_not_numpy_array():
    with pytest.raises(
        ValueError, match="Scores should be numpy array, got <class 'list'>."
    ):
        create_detection_message(
            np.array([[1.0, 2.0, 3.0, 4.0]]), [0.1, 0.2, 0.3], None, None
        )


def test_scores_shape():
    with pytest.raises(
        ValueError,
        match=re.escape("Scores should be of shape (N,) meaning, got (1, 1)."),
    ):
        create_detection_message(
            np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([[0.1]]), None, None
        )


def test_scores_length():
    with pytest.raises(
        ValueError,
        match=re.escape("Scores should have same length as bboxes, got 1 and 2."),
    ):
        create_detection_message(
            np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            np.array([0.1]),
            None,
            None,
        )


def test_labels_list():
    with pytest.raises(ValueError, match="Labels should be list, got <class 'int'>."):
        create_detection_message(
            np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([0.1]), 1, None
        )


def test_labels_bbox_lengths():
    with pytest.raises(
        ValueError, match="Labels should have same length as bboxes, got 1 and 2."
    ):
        create_detection_message(
            np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            np.array([0.1, 0.3]),
            [1],
            None,
        )


def test_in_label_elements():
    with pytest.raises(
        ValueError, match="Labels should be list of integers, got <class 'str'>."
    ):
        create_detection_message(
            np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([0.1]), ["1"], None
        )


def test_keypoints_list():
    with pytest.raises(
        ValueError, match="Keypoints should be list, got <class 'int'>."
    ):
        create_detection_message(
            np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([0.1]), None, 1
        )


def test_keypoints_bbox_lengths():
    with pytest.raises(
        ValueError, match="Keypoints should have same length as bboxes, got 1 and 2."
    ):
        create_detection_message(
            np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            np.array([0.1, 0.3]),
            None,
            [[(0, 0), (1, 1)]],
        )


def test_only_bboxes_scores():
    bboxes = np.array(
        [[20.0, 20.0, 40.0, 40.0], [50.0, 50.0, 100.0, 100.0], [10.0, 10.0, 20.0, 20.0]]
    )
    scores = np.array([0.1, 0.2, 0.3])

    message = create_detection_message(bboxes, scores, None, None)

    assert isinstance(message, dai.ImgDetections)
    assert all(
        isinstance(detection, dai.ImgDetection) for detection in message.detections
    )
    assert all(detection.label == 0 for detection in message.detections)
    for i, detection in enumerate(message.detections):
        assert detection.xmin == bboxes[i, 0]
        assert detection.ymin == bboxes[i, 1]
        assert detection.xmax == bboxes[i, 2]
        assert detection.ymax == bboxes[i, 3]
        assert np.isclose(detection.confidence, scores[i])


def test_bboxes_scores_labels():
    bboxes = np.array(
        [[20.0, 20.0, 40.0, 40.0], [50.0, 50.0, 100.0, 100.0], [10.0, 10.0, 20.0, 20.0]]
    )
    scores = np.array([0.1, 0.2, 0.3])
    labels = [1, 2, 3]

    message = create_detection_message(bboxes, scores, labels, None)

    for i, label in enumerate(labels):
        assert message.detections[i].label == label


def test_bboxes_scores_keypoints():
    bboxes = np.array(
        [[20.0, 20.0, 40.0, 40.0], [50.0, 50.0, 100.0, 100.0], [10.0, 10.0, 20.0, 20.0]]
    )
    scores = np.array([0.1, 0.2, 0.3])
    keypoints = [[(0, 0), (1, 1)], [], [(4, 4), (5, 5), (6, 6)]]

    message = create_detection_message(bboxes, scores, None, keypoints)

    assert isinstance(message, ImgDetectionsExtended)
    assert all(
        isinstance(detection, ImgDetectionExtended) for detection in message.detections
    )

    for i, detection in enumerate(message.detections):
        assert detection.keypoints == keypoints[i]
        assert detection.xmin == bboxes[i, 0]
        assert detection.ymin == bboxes[i, 1]
        assert detection.xmax == bboxes[i, 2]
        assert detection.ymax == bboxes[i, 3]
        assert np.isclose(detection.confidence, scores[i])
        assert detection.label == 0


if __name__ == "__main__":
    pytest.main()

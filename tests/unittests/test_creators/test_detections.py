import numpy as np
import pytest

from depthai_nodes.ml.messages import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.ml.messages.creators import (
    create_detection_message,
)


def test_valid_input():
    bboxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
    scores = np.array([0.9, 0.8])
    angles = np.array([45, -45])
    labels = np.array([1, 2])
    keypoints = np.array([[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3], [0.4, 0.4]]])
    keypoints_scores = np.array([[0.9, 0.8], [0.7, 0.6]])
    masks = np.array([[0, 1], [1, 0]])

    message = create_detection_message(
        bboxes, scores, angles, labels, keypoints, keypoints_scores, masks
    )

    assert isinstance(message, ImgDetectionsExtended)
    assert len(message.detections) == 2
    assert all(
        isinstance(detection, ImgDetectionExtended) for detection in message.detections
    )
    assert message.detections[0].confidence == 0.9
    assert message.detections[1].confidence == 0.8
    assert message.detections[0].label == 1
    assert message.detections[1].label == 2
    for i, detection in enumerate(message.detections):
        assert detection.confidence == scores[i]
        assert detection.label == labels[i]
        assert detection.rotated_rect.angle == angles[i]

        assert np.allclose(detection.rotated_rect.center.x, bboxes[i][0], atol=1e-3)
        assert np.allclose(detection.rotated_rect.center.y, bboxes[i][1], atol=1e-3)
        assert np.allclose(detection.rotated_rect.size.width, bboxes[i][2], atol=1e-3)
        assert np.allclose(detection.rotated_rect.size.height, bboxes[i][3], atol=1e-3)

        assert len(message.detections) == 2
        for j, keypoint in enumerate(detection.keypoints):
            assert keypoint.x == keypoints[i][j][0]
            assert keypoint.y == keypoints[i][j][1]
            assert keypoint.confidence == keypoints_scores[i][j]

    assert np.allclose(message.masks, masks, atol=1e-3)


def test_valid_input_no_optional():
    bboxes = np.array([[0.5, 0.5, 0.2, 0.2]])
    scores = np.array([0.9])
    message = create_detection_message(bboxes, scores)

    assert isinstance(message, ImgDetectionsExtended)
    assert len(message.detections) == 1
    assert message.detections[0].confidence == 0.9


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
    keypoints = np.array(
        [[(0.0, 0.0), (0.1, 0.1)], [(0.2, 0.2), (0.3, 0.3)], [(0.4, 0.4), (0.5, 0.5)]]
    )

    message = create_detection_message(
        bboxes=bboxes, scores=scores, keypoints=keypoints
    )

    assert isinstance(message, ImgDetectionsExtended)

    for i, detection in enumerate(message.detections):
        for ii, keypoint in enumerate(detection.keypoints):
            assert keypoint.x == keypoints[i][ii][0]
            assert keypoint.y == keypoints[i][ii][1]


def test_bboxes_masks():
    bboxes = np.array(
        [[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2]]
    )

    scores = np.array([0.1, 0.2, 0.3])
    mask = np.array([[-1, -1, -1], [-1, 0, -1], [-1, -1, -1]], dtype=np.int16)

    message = create_detection_message(bboxes=bboxes, scores=scores, masks=mask)

    assert isinstance(message, ImgDetectionsExtended)
    assert np.array_equal(message.masks, mask)


def test_no_bboxes_masks():
    bboxes = np.array([])
    scores = np.array([])
    mask = np.array([[-1, -1, -1], [-1, 0, -1], [-1, -1, -1]], dtype=np.int16)

    message = create_detection_message(bboxes=bboxes, scores=scores, masks=mask)

    assert isinstance(message, ImgDetectionsExtended)
    assert np.array_equal(message.masks, mask)


def test_no_masks_no_bboxes():
    bboxes = np.array([])
    scores = np.array([])
    mask = np.array([[]], dtype=np.int16)

    message = create_detection_message(bboxes=bboxes, scores=scores, masks=mask)

    assert isinstance(message, ImgDetectionsExtended)
    assert np.array_equal(message.masks, mask)


def test_empty_bboxes():
    bboxes = np.array([])
    scores = np.array([])
    message = create_detection_message(bboxes, scores)

    assert isinstance(message, ImgDetectionsExtended)
    assert len(message.detections) == 0


def test_invalid_bboxes_type():
    with pytest.raises(ValueError):
        create_detection_message([[0.5, 0.5, 0.2, 0.2]], np.array([0.9]))


def test_invalid_bboxes_shape():
    with pytest.raises(ValueError):
        create_detection_message(np.array([0.5, 0.5, 0.2, 0.2]), np.array([0.9]))


def test_invalid_scores_type():
    with pytest.raises(ValueError):
        create_detection_message(np.array([[0.5, 0.5, 0.2, 0.2]]), [0.9])


def test_invalid_scores_shape():
    with pytest.raises(ValueError):
        create_detection_message(np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([[0.9]]))


def test_mismatched_bboxes_scores_length():
    with pytest.raises(ValueError):
        create_detection_message(np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([0.9, 0.8]))


def test_invalid_labels_type():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([0.9]), labels=[1]
        )


def test_mismatched_bboxes_labels_length():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([0.9]), labels=np.array([1, 2])
        )


def test_invalid_angles_type():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([0.9]), angles=[45]
        )


def test_mismatched_bboxes_angles_length():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]),
            np.array([0.9]),
            angles=np.array([45, -45]),
        )


def test_invalid_angles_range():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([0.9]), angles=np.array([400])
        )


def test_invalid_keypoints_type():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([0.9]), keypoints=[[[0.1, 0.1]]]
        )


def test_invalid_keypoints_shape():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]),
            np.array([0.9]),
            keypoints=np.array([0.1, 0.1]),
        )


def test_mismatched_bboxes_keypoints_length():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]),
            np.array([0.9]),
            keypoints=np.array([[[0.1, 0.1]], [[0.2, 0.2]]]),
        )


def test_invalid_keypoints_dim():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]),
            np.array([0.9]),
            keypoints=np.array([[[0.1, 0.1, 0.1, 0.1]]]),
        )


def test_invalid_keypoints_scores_type():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]),
            np.array([0.9]),
            keypoints=np.array([[[0.1, 0.1]]]),
            keypoints_scores=[[[0.9]]],
        )


def test_mismatched_keypoints_keypoints_scores_length():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]),
            np.array([0.9]),
            keypoints=np.array([[[0.1, 0.1]]]),
            keypoints_scores=np.array([[[0.9], [0.8]]]),
        )


def test_invalid_keypoints_scores_range():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]),
            np.array([0.9]),
            keypoints=np.array([[[0.1, 0.1]]]),
            keypoints_scores=np.array([[[1.1]]]),
        )


def test_invalid_masks_type():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([0.9]), masks=[[0, 1]]
        )


def test_invalid_masks_shape():
    with pytest.raises(ValueError):
        create_detection_message(
            np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([0.9]), masks=np.array([0, 1])
        )

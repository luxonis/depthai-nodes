import numpy as np
import pytest

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.message.creators import (
    create_detection_message,
)

BBOXES = np.array([[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2]])
SCORES = np.array([0.1, 0.2, 0.3])
LABELS = np.array([1, 2, 3])
ANGLES = np.array([45, -45, 0])
KPTS = np.array(
    [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3], [0.4, 0.4]], [[0.5, 0.5], [0.5, 0.5]]]
)
KPTS_SCORES = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
MASKS = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int16)

ONE_BBOX = np.array([[0.5, 0.5, 0.2, 0.2]])
ONE_SCORE = np.array([0.9])


def test_valid_input():
    message = create_detection_message(
        bboxes=BBOXES,
        scores=SCORES,
        labels=LABELS,
        angles=ANGLES,
        keypoints=KPTS,
        keypoints_scores=KPTS_SCORES,
        masks=MASKS,
    )

    assert isinstance(message, ImgDetectionsExtended)
    assert len(message.detections) == 3

    for i, detection in enumerate(message.detections):
        assert isinstance(detection, ImgDetectionExtended)
        assert detection.confidence == SCORES[i]
        assert detection.label == LABELS[i]
        assert detection.rotated_rect.angle == ANGLES[i]

        assert np.allclose(detection.rotated_rect.center.x, BBOXES[i][0], atol=1e-3)
        assert np.allclose(detection.rotated_rect.center.y, BBOXES[i][1], atol=1e-3)
        assert np.allclose(detection.rotated_rect.size.width, BBOXES[i][2], atol=1e-3)
        assert np.allclose(detection.rotated_rect.size.height, BBOXES[i][3], atol=1e-3)

        for j, keypoint in enumerate(detection.keypoints):
            assert keypoint.x == KPTS[i][j][0]
            assert keypoint.y == KPTS[i][j][1]
            assert keypoint.confidence == KPTS_SCORES[i][j]

    assert np.allclose(message.masks, MASKS, atol=1e-3)


def test_valid_input_no_optional():
    message = create_detection_message(BBOXES, SCORES)

    assert isinstance(message, ImgDetectionsExtended)
    assert len(message.detections) == 3

    for i, detection in enumerate(message.detections):
        assert isinstance(detection, ImgDetectionExtended)
        assert detection.confidence == SCORES[i]


def test_bboxes_scores_labels():
    message = create_detection_message(bboxes=BBOXES, scores=SCORES, labels=LABELS)

    for i, label in enumerate(LABELS):
        assert message.detections[i].label == label


def test_bboxes_scores_keypoints():
    message = create_detection_message(bboxes=BBOXES, scores=SCORES, keypoints=KPTS)

    assert isinstance(message, ImgDetectionsExtended)

    for i, detection in enumerate(message.detections):
        for ii, keypoint in enumerate(detection.keypoints):
            assert keypoint.x == KPTS[i][ii][0]
            assert keypoint.y == KPTS[i][ii][1]


def test_bboxes_masks():
    message = create_detection_message(bboxes=BBOXES, scores=SCORES, masks=MASKS)

    assert isinstance(message, ImgDetectionsExtended)
    assert np.array_equal(message.masks, MASKS)


def test_no_bboxes_masks():
    bboxes = np.array([])
    scores = np.array([])

    message = create_detection_message(bboxes=bboxes, scores=scores, masks=MASKS)

    assert isinstance(message, ImgDetectionsExtended)
    assert np.array_equal(message.masks, MASKS)


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
        create_detection_message(BBOXES.tolist(), SCORES)


def test_invalid_bboxes_shape():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX[0], ONE_SCORE)


def test_invalid_scores_type():
    with pytest.raises(ValueError):
        create_detection_message(BBOXES, SCORES.tolist())


def test_invalid_scores_shape():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, np.array([ONE_SCORE]))


def test_mismatched_bboxes_scores_length():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, np.array([0.9, 0.8]))


def test_invalid_labels_type():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, ONE_SCORE, labels=[1])


def test_mismatched_bboxes_labels_length():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, ONE_SCORE, labels=np.array([1, 2]))


def test_invalid_angles_type():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, ONE_SCORE, angles=[45])


def test_mismatched_bboxes_angles_length():
    with pytest.raises(ValueError):
        create_detection_message(
            ONE_BBOX,
            ONE_SCORE,
            angles=np.array([45, -45]),
        )


def test_invalid_angles_range():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, ONE_SCORE, angles=np.array([400]))


def test_invalid_keypoints_type():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, ONE_SCORE, keypoints=[[[0.1, 0.1]]])


def test_invalid_keypoints_shape():
    with pytest.raises(ValueError):
        create_detection_message(
            ONE_BBOX,
            ONE_SCORE,
            keypoints=np.array([0.1, 0.1]),
        )


def test_mismatched_bboxes_keypoints_length():
    with pytest.raises(ValueError):
        create_detection_message(
            ONE_BBOX,
            ONE_SCORE,
            keypoints=np.array([[[0.1, 0.1]], [[0.2, 0.2]]]),
        )


def test_invalid_keypoints_dim():
    with pytest.raises(ValueError):
        create_detection_message(
            ONE_BBOX,
            ONE_SCORE,
            keypoints=np.array([[[0.1, 0.1, 0.1, 0.1]]]),
        )


def test_invalid_keypoints_scores_type():
    with pytest.raises(ValueError):
        create_detection_message(
            ONE_BBOX,
            ONE_SCORE,
            keypoints=np.array([[[0.1, 0.1]]]),
            keypoints_scores=[[[0.9]]],
        )


def test_mismatched_keypoints_keypoints_scores_length():
    with pytest.raises(ValueError):
        create_detection_message(
            ONE_BBOX,
            ONE_SCORE,
            keypoints=np.array([[[0.1, 0.1]]]),
            keypoints_scores=np.array([[[0.9], [0.8]]]),
        )


def test_invalid_keypoints_scores_range():
    with pytest.raises(ValueError):
        create_detection_message(
            ONE_BBOX,
            ONE_SCORE,
            keypoints=np.array([[[0.1, 0.1]]]),
            keypoints_scores=np.array([[[1.1]]]),
        )


def test_invalid_masks_type():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, ONE_SCORE, masks=[[0, 1]])


def test_invalid_masks_shape():
    with pytest.raises(ValueError):
        create_detection_message(ONE_BBOX, ONE_SCORE, masks=np.array([0, 1]))

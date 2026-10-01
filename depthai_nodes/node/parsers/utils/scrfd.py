import numpy as np

from depthai_nodes.node.parsers.utils.bbox_format_converters import xyxy_to_xywh
from depthai_nodes.node.parsers.utils.nms import nms


def compute_anchor_centers(
    strides: list[int], input_size: tuple[int, int], num_anchors: int
) -> dict[int, np.ndarray]:
    """Compute the anchor centers for a given list of strides, input size, and number of
    anchors.

    Args:
        strides: List of strides.
        input_size: Input size.
        num_anchors: Number of anchors.

    Returns:
        Dictionary of anchor centers.
    """
    anchor_centers_dict = {}
    for stride in strides:
        height = input_size[0] // stride
        width = input_size[1] // stride
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
            np.float32
        )
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape(
                (-1, 2)
            )
        anchor_centers_dict[stride] = anchor_centers
    return anchor_centers_dict


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (``np.ndarray``): Shape (n, 2), [x, y].
        distance (``np.ndarray``): Distance from the given point to 4 boundaries (left,
            top, right, bottom).
        max_shape (``tuple[int, int]``): Shape of the image.

    Returns:
        ``np.ndarray``: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints.

    Args:
        points (``np.ndarray``): Shape (n, 2), [x, y].
        distance (``np.ndarray``): Distance from the given point to 4 boundaries (left,
            top, right, bottom).
        max_shape (``tuple[int, int]``): Shape of the image.

    Returns:
        ``np.ndarray``: Decoded keypoints.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def decode_scrfd(
    bboxes_concatenated,
    scores_concatenated,
    kps_concatenated,
    feat_stride_fpn,
    input_size,
    num_anchors,
    score_threshold,
    nms_threshold,
    anchors,
):
    """Decode the detection results of SCRFD.

    Args:
        bboxes_concatenated (``list[np.ndarray]``): List of bounding box predictions for
            each scale.
        scores_concatenated (``list[np.ndarray]``): List of confidence score predictions
            for each scale.
        kps_concatenated (``list[np.ndarray]``): List of keypoint predictions for each
            scale.
        feat_stride_fpn (``list[int]``): List of feature strides for each scale.
        input_size (``tuple[int]``): Input size of the model.
        num_anchors (``int``): Number of anchors.
        score_threshold (``float``): Confidence score threshold.
        nms_threshold (``float``): Non-maximum suppression threshold.
        anchors (``dict[int, np.ndarray]``): Dictionary of anchors.

    Returns:
        ``tuple[np.ndarray, np.ndarray, np.ndarray]``: Bounding boxes, confidence
            scores, and keypoints of detected objects.
    """
    scores_list = []
    bboxes_list = []
    kps_list = []

    for idx, stride in enumerate(feat_stride_fpn):
        scores = scores_concatenated[idx]
        bbox_preds = bboxes_concatenated[idx] * stride
        kps_preds = kps_concatenated[idx] * stride

        height = input_size[0] // stride
        width = input_size[1] // stride

        anchor_centers = anchors[stride]

        pos_inds = np.where(scores >= score_threshold)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores.reshape(-1, 1))
        bboxes_list.append(pos_bboxes)

        kpss = distance2kps(anchor_centers, kps_preds)
        kpss = kpss.reshape((kpss.shape[0], -1, 2))
        pos_kpss = kpss[pos_inds]
        kps_list.append(pos_kpss)

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes = np.vstack(bboxes_list)
    kpss = np.vstack(kps_list)

    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det, nms_threshold)
    det = pre_det[keep, :]
    kpss = kpss[order, :, :]
    kpss = kpss[keep, :, :]

    height, width = input_size
    scores = det[:, 4]
    bboxes = det[:, :4] / np.array([width, height] * 2)

    keypoints = kpss / np.tile([width, height], (5, 1))
    keypoints = keypoints.reshape(-1, 5, 2)
    keypoints = np.clip(keypoints, 0, 1)

    return bboxes, scores, keypoints


def compute_scrfd_detections(
    *,
    bboxes_concatenated: list[np.ndarray],
    scores_concatenated: list[np.ndarray],
    kps_concatenated: list[np.ndarray],
    feat_stride_fpn: tuple[int, ...] | list[int],
    input_size: tuple[int, int],
    num_anchors: int,
    score_threshold: float,
    nms_threshold: float,
    anchors: dict[int, np.ndarray],
    label_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """Decode SCRFD outputs into final detection payloads."""
    bboxes, scores, keypoints = decode_scrfd(
        bboxes_concatenated=bboxes_concatenated,
        scores_concatenated=scores_concatenated,
        kps_concatenated=kps_concatenated,
        feat_stride_fpn=feat_stride_fpn,
        input_size=input_size,
        num_anchors=num_anchors,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        anchors=anchors,
    )
    bboxes = np.clip(bboxes, 0, 1)
    bboxes = xyxy_to_xywh(bboxes)
    bboxes = np.clip(bboxes, 0, 1)

    labels = np.zeros(len(bboxes), dtype=int)
    mapped_label_names = (
        [label_names[label] for label in labels] if label_names else None
    )
    return bboxes, scores, keypoints, labels, mapped_label_names

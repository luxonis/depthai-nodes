import numpy as np


def nms(dets, nms_thresh=0.5):
    """Non-maximum suppression.

    @param dets: Bounding boxes and confidence scores.
    @type dets: np.ndarray
    @return: Indices of the detections to keep.
    @rtype: list[int]
    """
    thresh = nms_thresh
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    @param points: Shape (n, 2), [x, y].
    @type points: np.ndarray
    @param distance: Distance from the given point to 4 boundaries (left, top, right,
        bottom).
    @type distance: np.ndarray
    @param max_shape: Shape of the image.
    @type max_shape: Tuple[int, int]
    @return: Decoded bboxes.
    @rtype: np.ndarray
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

    @param points: Shape (n, 2), [x, y].
    @type points: np.ndarray
    @param distance: Distance from the given point to 4 boundaries (left, top, right,
        bottom).
    @type distance: np.ndarray
    @param max_shape: Shape of the image.
    @type max_shape: Tuple[int, int]
    @return: Decoded keypoints.
    @rtype: np.ndarray
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
):
    """Decode the detection results of SCRFD.

    @param bboxes_concatenated: List of bounding box predictions for each scale.
    @type bboxes_concatenated: list[np.ndarray]
    @param scores_concatenated: List of confidence score predictions for each scale.
    @type scores_concatenated: list[np.ndarray]
    @param kps_concatenated: List of keypoint predictions for each scale.
    @type kps_concatenated: list[np.ndarray]
    @param feat_stride_fpn: List of feature strides for each scale.
    @type feat_stride_fpn: list[int]
    @param input_size: Input size of the model.
    @type input_size: tuple[int]
    @param num_anchors: Number of anchors.
    @type num_anchors: int
    @param score_threshold: Confidence score threshold.
    @type score_threshold: float
    @param nms_threshold: Non-maximum suppression threshold.
    @type nms_threshold: float
    @return: Bounding boxes, confidence scores, and keypoints of detected objects.
    @rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
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

        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
            np.float32
        )
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape(
                (-1, 2)
            )

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

    scores = det[:, 4]
    bboxes = np.int32(det[:, :4])
    keypoints = np.int32(kpss)
    keypoints = keypoints.reshape(-1, 5, 2)

    return bboxes, scores, keypoints

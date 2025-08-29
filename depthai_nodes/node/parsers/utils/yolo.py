import time
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from depthai_nodes.logging import get_logger
from depthai_nodes.node.parsers.utils import sigmoid, xywh_to_xyxy
from depthai_nodes.node.parsers.utils.nms import nms

logger = get_logger(__name__)


class YOLOSubtype(str, Enum):
    V3 = "yolov3"
    V3T = "yolov3-tiny"
    V3U = "yolov3-u"
    V3UT = "yolov3-tinyu"
    V4 = "yolov4"
    V4T = "yolov4-tiny"
    V5 = "yolov5"
    V5U = "yolov5-u"
    V6 = "yolov6"
    V6R1 = "yolov6r1"
    V6R2 = "yolov6r2"  # NOTE: used by DAI but internally same as V6
    V7 = "yolov7"
    V8 = "yolov8"
    V9 = "yolov9"
    V10 = "yolov10"
    P = "yolo-p"
    GOLD = "yolo-gold"
    DEFAULT = ""


def make_grid_numpy(ny: int, nx: int, na: int) -> np.ndarray:
    """Create a grid of shape (1, na, ny, nx, 2)

    @param ny: Number of y coordinates.
    @type ny: int
    @param nx: Number of x coordinates.
    @type nx: int
    @param na: Number of anchors.
    @type na: int
    @return: Grid.
    @rtype: np.ndarray
    """
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.stack((xv, yv), 2).reshape(1, na, ny, nx, 2)


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    classes: Optional[List] = None,
    num_classes: int = 1,
    agnostic: bool = False,
    multi_label: bool = False,
    max_det: int = 300,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    kpts_mode: bool = False,
    det_mode: bool = False,
) -> List[np.ndarray]:
    """Performs Non-Maximum Suppression (NMS) on inference results.

    @param prediction: Prediction from the model, shape = (batch_size, boxes, xy+wh+...)
    @type prediction: np.ndarray
    @param conf_thres: Confidence threshold.
    @type conf_thres: float
    @param iou_thres: Intersection over union threshold.
    @type iou_thres: float
    @param classes: For filtering by classes.
    @type classes: Optional[List]
    @param num_classes: Number of classes.
    @type num_classes: int
    @param agnostic: Runs NMS on all boxes together rather than per class if True.
    @type agnostic: bool
    @param multi_label: Multilabel classification.
    @type multi_label: bool
    @param max_det: Limiting detections.
    @type max_det: int
    @param max_time_img: Maximum time for processing an image.
    @type max_time_img: float
    @param max_nms: Maximum number of boxes.
    @type max_nms: int
    @param max_wh: Maximum width and height.
    @type max_wh: int
    @param kpts_mode: Keypoints mode.
    @type kpts_mode: bool
    @param det_mode: Detection only mode. If True, the output will only contain bbox detections.
    @type det_mode: bool
    @return: An array of detections. If det_mode is False, the detections may include kpts or segmentation outputs.
    @rtype: List[np.ndarray]
    """
    bs = prediction.shape[0]  # batch size

    offset = prediction.shape[2] - num_classes
    num_classes_check = prediction.shape[2] - offset

    nm = prediction.shape[2] - num_classes - 5
    pred_candidates = prediction[..., 4] > conf_thres  # candidates

    # Check the parameters.
    if num_classes != num_classes_check:
        raise ValueError(
            f"Number of classes {num_classes} does not match the model {num_classes_check}."
        )
    if conf_thres < 0 or conf_thres > 1:
        raise ValueError(
            f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0."
        )
    if iou_thres < 0 or iou_thres > 1:
        raise ValueError(
            f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0."
        )

    # Function settings.
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [np.zeros((0, 6 + nm))] * prediction.shape[0]

    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh_to_xyxy(x[:, :4])
        cls = x[:, 5 : 5 + num_classes]
        other = x[:, 5 + num_classes :]  # Either kpts or pos

        if multi_label:
            box_idx, class_idx = (cls > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate(
                (
                    box[box_idx],
                    x[box_idx, class_idx + 5, None],
                    class_idx[:, None],
                    other[box_idx, :],
                ),
                1,
            )
        else:  # Only keep the class with highest scores.
            class_idx = np.expand_dims(cls.argmax(1), 1)
            conf = cls.max(1, keepdims=True)
            x = np.concatenate((box, conf, class_idx, other), 1)[
                conf.flatten() > conf_thres
            ]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort()[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = (
            x[:, :4] + class_offset,
            x[:, 4][..., np.newaxis],
        )  # boxes (offset by class), scores
        keep_box_idx = np.array(
            nms(np.hstack((boxes, scores)).astype(np.float32, copy=False), iou_thres)
        )

        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            logger.info(f"NMS cost time exceed the limited {time_limit}s.")
            break  # time limit exceeded

    return output


def parse_yolo_output(
    out: np.ndarray,
    stride: int,
    num_outputs: int,
    anchors: Optional[np.ndarray] = None,
    head_id: int = -1,
    kpts: Optional[np.ndarray] = None,
    det_mode: bool = False,
    subtype: YOLOSubtype = YOLOSubtype.DEFAULT,
) -> np.ndarray:
    """Parse a single channel output of an YOLO model.

    @param out: A single output of an YOLO model for the given channel.
    @type out: np.ndarray
    @param stride: Stride.
    @type stride: int
    @param num_outputs: Number of outputs of the model.
    @type num_outputs: int
    @param anchors: Anchors for the given head.
    @type anchors: Optional[np.ndarray]
    @param head_id: Head ID.
    @type head_id: int
    @param kpts: A single output of keypoints for the given channel.
    @type kpts: Optional[np.ndarray]
    @param det_mode: Detection only mode.
    @type det_mode: bool
    @param subtype: YOLO version.
    @type subtype: YOLOSubtype
    @return: Parsed output.
    @rtype: np.ndarray
    """
    na = (
        anchors.shape[0] // 2 if anchors is not None else 1
    )  # number of anchors per head
    bs, _, ny, nx = out.shape  # bs - batch size, ny|nx - y and x of grid cells

    if subtype in [
        YOLOSubtype.P,
        YOLOSubtype.V5,
        YOLOSubtype.V5U,
        YOLOSubtype.V7,
        YOLOSubtype.V3,
        YOLOSubtype.V3T,
        YOLOSubtype.V4,
        YOLOSubtype.V4T,
    ]:
        grid = make_grid_numpy(ny, nx, 1)
    else:
        grid = make_grid_numpy(ny, nx, na)

    out = out.reshape(bs, na, num_outputs, ny, nx).transpose((0, 1, 3, 4, 2))

    if anchors is not None:
        if isinstance(anchors, np.ndarray):
            anchors = anchors.reshape(bs, -1, 1, 1, 2)
        else:
            anchors = np.array(anchors).reshape(bs, -1, 1, 1, 2)
        assert (
            anchors.shape[1] == na
        ), f"Anchor shape mismatch at dimension 1: {anchors.shape[1]} vs {na}"

        if subtype in [YOLOSubtype.V3, YOLOSubtype.V3T]:
            c_xy = out[..., 0:2] + grid
            wh = np.exp(out[..., 2:4])
        elif subtype in [YOLOSubtype.V4, YOLOSubtype.V4T]:
            raise NotImplementedError("YOLOv4 is not supported yet")
        else:
            c_xy = out[..., 0:2] * 2 - 0.5 + grid
            wh = (out[..., 2:4] * 2) ** 2

        out[..., 0:2] = c_xy * stride
        out[..., 2:4] = wh * anchors
    else:
        if subtype == YOLOSubtype.V6R1:
            c_xy = out[..., 0:2] + grid
            wh = np.exp(out[..., 2:4])
        else:
            x1y1 = grid - out[..., 0:2] + 0.5
            x2y2 = grid + out[..., 2:4] + 0.5
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
        out[..., 0:2] = c_xy * stride
        out[..., 2:4] = wh * stride

    if det_mode:
        # Detection
        out = out.reshape(bs, -1, num_outputs)
    elif kpts is None:
        # Segmentation
        x_coors = np.tile(np.arange(0, nx), (ny, 1))
        x_coors = np.repeat(x_coors[np.newaxis, np.newaxis, ..., np.newaxis], 1, axis=1)

        y_coors = np.tile(np.arange(0, ny)[np.newaxis, ...].T, (1, nx))
        y_coors = np.repeat(y_coors[np.newaxis, np.newaxis, ..., np.newaxis], 1, axis=1)

        ai = (
            np.ones((bs, na, ny, nx))
            * np.arange(na)[np.newaxis, ..., np.newaxis, np.newaxis]
        )
        ai = ai[..., np.newaxis]
        hi = np.ones((bs, na, ny, nx, 1)) * head_id

        out = np.concatenate((out, hi, ai, x_coors, y_coors), axis=4).reshape(
            bs, na * ny * nx, -1
        )
    else:
        # Keypoints

        # NOTE: For now we omit "guessing" if sigmoid is applied, should be better handled with flags in NNarchive/Parser
        # sigmoid_applied = np.all((kpts[:, 2::3] >= 0) & (kpts[:, 2::3] <= 1))
        # if not sigmoid_applied:
        #     kpts[:, 2::3] = sigmoid(kpts[:, 2::3])

        kpts[:, 2::3] = sigmoid(kpts[:, 2::3])
        kpts_out = kpts.transpose(0, 2, 1)
        out = out.reshape(bs, ny * nx, -1)
        out = np.concatenate((out, kpts_out), axis=2)

    return out


def parse_kpts(
    kpts: np.ndarray, n_keypoints: int, img_shape: Tuple[int, int]
) -> List[Tuple[int, int, float]]:
    """Parse keypoints.

    @param kpts: Result keypoints.
    @type kpts: np.ndarray
    @param n_keypoints: Number of keypoints.
    @type n_keypoints: int
    @param img_shape: Image shape of the model input in (height, width) format.
    @type img_shape: Tuple[int, int]
    @return: Parsed keypoints.
    @rtype: List[Tuple[int, int, float]]
    """
    h, w = img_shape
    kps = []
    ndim = len(kpts) // n_keypoints
    for idx in range(0, kpts.shape[0], ndim):
        x, y = kpts[idx] / w, kpts[idx + 1] / h
        conf = kpts[idx + 2] if ndim == 3 else 1.0
        kps.append((x, y, conf))
    return kps


def decode_yolo_output(
    yolo_outputs: List[np.ndarray],
    strides: List[int],
    anchors: Optional[np.ndarray] = None,
    kpts: Optional[List[np.ndarray]] = None,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    num_classes: int = 1,
    det_mode: bool = False,
    subtype: YOLOSubtype = YOLOSubtype.DEFAULT,
    max_nms: int = 3000,
) -> np.ndarray:
    """Decode the output of an YOLO instance segmentation or pose estimation model.

    @param yolo_outputs: List of YOLO outputs.
    @type yolo_outputs: List[np.ndarray]
    @param strides: List of strides.
    @type strides: List[int]
    @param anchors: An optional array of anchors.
    @type anchors: Optional[np.ndarray]
    @param kpts: An optional list of keypoints.
    @type kpts: Optional[List[np.ndarray]]
    @param conf_thres: Confidence threshold.
    @type conf_thres: float
    @param iou_thres: Intersection over union threshold.
    @type iou_thres: float
    @param num_classes: Number of classes.
    @type num_classes: int
    @param det_mode: Detection only mode. If True, the output will only contain bbox
        detections.
    @type det_mode: bool
    @param subtype: YOLO version.
    @type subtype: YOLOSubtype
    @param max_nms: Maximum number of boxes to keep after NMS.
    @type max_nms: int
    @return: NMS output.
    @rtype: np.ndarray
    """
    num_outputs = num_classes + 5

    # 1. Parse and concatenate all head outputs efficiently
    filtered_outputs = []
    for i, (out_head, stride) in enumerate(zip(yolo_outputs, strides)):
        kpt = kpts[i] if kpts else None
        anchors_head = anchors[i] if anchors is not None else None
        out = parse_yolo_output(
            out_head,
            stride,
            num_outputs,
            anchors_head,
            head_id=i,
            kpts=kpt,
            det_mode=det_mode,
            subtype=subtype,
        )
        # Early filter: keep only predictions with objectness > conf_thres
        obj_scores = out[..., 4]
        mask = obj_scores > conf_thres
        if np.any(mask):
            filtered_outputs.append(out[mask])

    if not filtered_outputs:
        return np.zeros((0, num_outputs))  # no detections

    # 2. Concatenate all kept candidates at once
    output = np.concatenate(filtered_outputs, axis=0)
    if not np.unique(output[:, 4]).size == 1 and output.shape[0] > max_nms:
        idx = np.argsort(output[:, 4])[::-1][:max_nms]  # sort by objectness
        output = output[idx]

    # 3. Run NMS on filtered predictions (single batch)
    output_nms = non_max_suppression(
        output[None, ...],  # batch dim
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        num_classes=num_classes,
        max_nms=max_nms,
        det_mode=det_mode,
    )[0]

    return output_nms

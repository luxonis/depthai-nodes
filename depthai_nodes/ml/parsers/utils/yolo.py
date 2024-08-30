import time
from typing import List, Optional, Tuple

import numpy as np

from .nms import nms


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


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Converts (center x, center y, width, height) to (x1, y1, x2, y2).

    @param x: Bounding box.
    @type x: np.ndarray
    @return: Converted bounding box.
    @rtype: np.ndarray
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    classes: List = None,
    num_classes: int = 1,
    agnostic: bool = False,
    multi_label: bool = False,
    max_det: int = 300,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    kpts_mode: bool = False,
) -> List[np.ndarray]:
    """Performs Non-Maximum Suppression (NMS) on inference results.

    @param prediction: Prediction from the model, shape = (batch_size, boxes, xy+wh+...)
    @type prediction: np.ndarray
    @param conf_thres: Confidence threshold.
    @type conf_thres: float
    @param iou_thres: Intersection over union threshold.
    @type iou_thres: float
    @param classes: For filtering by classes.
    @type classes: List
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
    @return: An array of detections with either kpts or segmentation outputs.
    @rtype: List[np.ndarray]
    """
    bs = prediction.shape[0]  # batch size
    # Keypoints: 4 (bbox) + 1 (objectness) + 51 (kpts) = 56
    # Segmentation: 4 (bbox) + 1 (objectness) + 4 (pos) = 9
    num_classes_check = prediction.shape[2] - (
        56 if kpts_mode else 9
    )  # number of classes
    nm = prediction.shape[2] - num_classes - 5
    pred_candidates = prediction[..., 4] > conf_thres  # candidates

    # Check the parameters.
    assert (
        num_classes == num_classes_check
    ), f"Number of classes {num_classes} does not match the model {num_classes_check}"
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

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
        box = xywh2xyxy(x[:, :4])
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
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

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
            print(f"WARNING: NMS cost time exceed the limited {time_limit}s.")
            break  # time limit exceeded

    return output


def parse_yolo_outputs(
    outputs: List[np.ndarray],
    strides: List[int],
    anchors: np.ndarray,
    kpts: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Parse all outputs of an YOLO model (all channels).

    @param outputs: List of outputs of an YOLO model.
    @type outputs: List[np.ndarray]
    @param strides: List of strides.
    @type strides: List[int]
    @param anchors: List of anchors.
    @type anchors: np.ndarray
    @param kpts: An optional list of keypoints for each output.
    @type kpts: Optional[List[np.ndarray]]
    @return: Parsed output.
    @rtype: np.ndarray
    """
    output = None

    for i, (x, s, a) in enumerate(zip(outputs, strides, anchors)):
        kpt = kpts[i] if kpts is not None else None
        out = parse_yolo_output(x, s, a, head_id=i, kpts=kpt)
        output = out if output is None else np.concatenate((output, out), axis=1)

    return output


def parse_yolo_output(
    out: np.ndarray,
    stride: int,
    anchors: np.ndarray,
    head_id: int = -1,
    kpts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Parse a single channel output of an YOLO model.

    @param out: A single output of an YOLO model for the given channel.
    @type out: np.ndarray
    @param stride: Stride.
    @type stride: int
    @param anchors: Anchors.
    @type anchors: np.ndarray
    @param head_id: Head ID.
    @type head_id: int
    @param kpts: A single output of keypoints for the given channel.
    @type kpts: np.ndarray
    @return: Parsed output.
    @rtype: np.ndarray
    """
    na = 1 if anchors is None else len(anchors)  # number of anchors per head
    bs, _, ny, nx = out.shape  # bs - batch size, ny|nx - y and x of grid cells

    grid = make_grid_numpy(ny, nx, na)

    out = out.reshape(bs, na, -1, ny, nx).transpose((0, 1, 3, 4, 2))

    x1y1 = grid - out[..., 0:2] + 0.5
    x2y2 = grid + out[..., 2:4] + 0.5

    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    out[..., 0:2] = c_xy * stride
    out[..., 2:4] = wh * stride

    if kpts is None:
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
        if (kpts.shape[1] // 17) == 3:
            kpts[:, 2::3] = sigmoid(kpts[:, 2::3])
        kpts_out = kpts.transpose(0, 2, 1)
        out = out.reshape(bs, ny * nx, -1)
        out = np.concatenate((out, kpts_out), axis=2)

    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function.

    @param x: Input tensor.
    @type x: np.ndarray
    @return: A result tensor after applying a sigmoid function on the given input.
    @rtype: np.ndarray
    """
    return 1 / (1 + np.exp(-x))


def process_single_mask(protos: np.ndarray, mask_coeff: np.ndarray, mask_conf: float):
    """Process a single mask.

    @param protos: Protos.
    @type protos: np.ndarray
    @param mask_coeff: Mask coefficient.
    @type mask_coeff: np.ndarray
    @param mask_conf: Mask confidence.
    @type mask_conf: float
    @return: Processed mask.
    @rtype: np.ndarray
    """
    mask = sigmoid(np.sum(protos * mask_coeff[..., np.newaxis, np.newaxis], axis=0))
    return (mask > mask_conf).astype(np.uint8)


def parse_kpts(kpts: np.ndarray, n_keypoints: int) -> List[Tuple[int, int, float]]:
    """Parse keypoints.

    @param kpts: Result keypoints.
    @type kpts: np.ndarray
    @param n_keypoints: Number of keypoints.
    @type n_keypoints: int
    @return: Parsed keypoints.
    @rtype: List[Tuple[int, int, float]]
    """
    kps = []
    ndim = len(kpts) // n_keypoints
    for idx in range(0, kpts.shape[0], ndim):
        x, y = kpts[idx], kpts[idx + 1]
        conf = kpts[idx + 2] if ndim == 3 else 1.0
        kps.append((int(x), int(y), conf))
    return kps


def decode_yolo_output(
    yolo_outputs: List[np.ndarray],
    strides: List[int],
    anchors: List[Optional[np.ndarray]],
    kpts: List[np.ndarray] = None,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    num_classes: int = 1,
) -> np.ndarray:
    """Decode the output of an YOLO instance segmentation or pose estimation model.

    @param yolo_outputs: List of YOLO outputs.
    @type yolo_outputs: List[np.ndarray]
    @param strides: List of strides.
    @type strides: List[int]
    @param anchors: List of anchors.
    @type anchors: List[Optional[np.ndarray]]
    @param kpts: List of keypoints.
    @type kpts: List[np.ndarray]
    @param conf_thres: Confidence threshold.
    @type conf_thres: float
    @param iou_thres: Intersection over union threshold.
    @type iou_thres: float
    @param num_classes: Number of classes.
    @type num_classes: int
    @return: NMS output.
    @rtype: np.ndarray
    """
    output = parse_yolo_outputs(yolo_outputs, strides, anchors, kpts)
    output_nms = non_max_suppression(
        output,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        num_classes=num_classes,
        kpts_mode=kpts is not None,
    )[0]

    return output_nms

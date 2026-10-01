"""mediapipe.py.

Description: This script contains utility functions for decoding the output of the
MediaPipe hand tracking model.

This script contains code that is based on or directly taken from a public GitHub
repository:
https://github.com/geaxgx/depthai_hand_tracker

Original code author(s): geaxgx

License: MIT License

Copyright (c) [2021] [geax]
"""

import math
from collections import namedtuple

import cv2
import numpy as np


class HandRegion:
    """Store a detected palm and its derived rotated region.

    Attributes:
        pd_score: Palm detection confidence.
        pd_box: Normalized ``[x, y, width, height]`` box in the square image.
        pd_kps: Normalized ``[x, y]`` palm keypoints in the square image.
        rect_x_center: Normalized rotated-rectangle center X coordinate.
        rect_y_center: Normalized rotated-rectangle center Y coordinate.
        rect_w: Normalized rectangle width, which may exceed 1.
        rect_h: Normalized rectangle height, which may exceed 1.
        rotation: Rectangle rotation relative to the Y axis, in radians.
        rect_x_center_a: Rectangle center X coordinate in square-image pixels.
        rect_y_center_a: Rectangle center Y coordinate in square-image pixels.
        rect_w_a: Rectangle width in square-image pixels.
        rect_h_a: Rectangle height in square-image pixels.
        rect_points: Four rectangle corners in pixels. Coordinates refer to the square
            image during processing and the source image on return.
    """

    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        """Store palm detection values before deriving a rotated region.

        Args:
            pd_score: Optional detection confidence.
            pd_box: Optional normalized ``[x, y, width, height]`` box.
            pd_kps: Optional normalized palm keypoints.
        """
        self.pd_score = pd_score  # Palm detection score
        self.pd_box = pd_box  # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps  # Palm detection keypoints


SSDAnchorOptions = namedtuple(
    "SSDAnchorOptions",
    [
        "num_layers",
        "min_scale",
        "max_scale",
        "input_size_height",
        "input_size_width",
        "anchor_offset_x",
        "anchor_offset_y",
        "strides",
        "aspect_ratios",
        "reduce_boxes_in_lowest_layer",
        "interpolated_scale_aspect_ratio",
        "fixed_anchor_size",
    ],
)


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    """Interpolate an anchor scale across feature strides.

    Args:
        min_scale: Scale at the first stride.
        max_scale: Scale at the last stride.
        stride_index: Zero-based stride index.
        num_strides: Number of strides; when one, use the midpoint scale.

    Returns:
        Interpolated anchor scale.
    """
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)


def generate_anchors(options):
    """Generate SSD anchors using MediaPipe's anchor layout.

    Based on the MediaPipe ``ssd_anchors_calculator.cc`` implementation.

    Args:
        options (``SSDAnchorOptions``): Layer sizes, strides, scales, and aspect ratios.

    Returns:
        Array of anchors in ``[x_center, y_center, width, height]`` format.
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (
            last_same_stride_layer < n_strides
            and options.strides[last_same_stride_layer] == options.strides[layer_id]
        ):
            scale = calculate_scale(
                options.min_scale, options.max_scale, last_same_stride_layer, n_strides
            )
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides - 1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(
                            options.min_scale,
                            options.max_scale,
                            last_same_stride_layer + 1,
                            n_strides,
                        )
                    scales.append(math.sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1

        for i, r in enumerate(aspect_ratios):
            ratio_sqrts = math.sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = math.ceil(options.input_size_height / stride)
        feature_map_width = math.ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    # new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                        # new_anchor.w = 1.0
                        # new_anchor.h = 1.0
                    else:
                        new_anchor = [
                            x_center,
                            y_center,
                            anchor_width[anchor_id],
                            anchor_height[anchor_id],
                        ]
                        # new_anchor.w = anchor_width[anchor_id]
                        # new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer
    return np.array(anchors)


def generate_handtracker_anchors(input_size_width, input_size_height):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
    """Generate anchors for the MediaPipe palm-detection layout.

    Args:
        input_size_width: Model input width in pixels.
        input_size_height: Model input height in pixels.

    Returns:
        Array of normalized center-XY/width/height anchors for strides 8, 16, 16, and
        16.
    """
    anchor_options = SSDAnchorOptions(
        num_layers=4,
        min_scale=0.1484375,
        max_scale=0.75,
        input_size_height=input_size_height,
        input_size_width=input_size_width,
        anchor_offset_x=0.5,
        anchor_offset_y=0.5,
        strides=[8, 16, 16, 16],
        aspect_ratios=[1.0],
        reduce_boxes_in_lowest_layer=False,
        interpolated_scale_aspect_ratio=1.0,
        fixed_anchor_size=True,
    )
    return generate_anchors(anchor_options)


def decode_bboxes(score_thresh, scores, bboxes, anchors, scale=128, best_only=False):
    # Wi, hi : NN input shape
    # mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc # Decodes
    # the detection tensors generated by the model, based on # the SSD anchors and the
    # specification in the options, into a vector of # detections. Each detection
    # describes a detected object.

    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt :
    # node {
    #     calculator: "TensorsToDetectionsCalculator"
    #     input_stream: "TENSORS:detection_tensors"
    #     input_side_packet: "ANCHORS:anchors"
    #     output_stream: "DETECTIONS:unfiltered_detections"
    #     options: {
    #         [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
    #         num_classes: 1
    #         num_boxes: 896
    #         num_coords: 18
    #         box_coord_offset: 0
    #         keypoint_coord_offset: 4
    #         num_keypoints: 7
    #         num_values_per_keypoint: 2
    #         sigmoid_score: true
    #         score_clipping_thresh: 100.0
    #         reverse_output_order: true

    #         x_scale: 128.0
    #         y_scale: 128.0
    #         h_scale: 128.0
    #         w_scale: 128.0
    #         min_score_thresh: 0.5
    #         }
    #     }
    # }
    # node {
    #     calculator: "TensorsToDetectionsCalculator"
    #     input_stream: "TENSORS:detection_tensors"
    #     input_side_packet: "ANCHORS:anchors"
    #     output_stream: "DETECTIONS:unfiltered_detections"
    #     options: {
    #         [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
    #         num_classes: 1
    #         num_boxes: 2016
    #         num_coords: 18
    #         box_coord_offset: 0
    #         keypoint_coord_offset: 4
    #         num_keypoints: 7
    #         num_values_per_keypoint: 2
    #         sigmoid_score: true
    #         score_clipping_thresh: 100.0
    #         reverse_output_order: true

    #         x_scale: 192.0
    #         y_scale: 192.0
    #         w_scale: 192.0
    #         h_scale: 192.0
    #         min_score_thresh: 0.5
    #         }
    #     }
    # }

    # scores: shape = [number of anchors 896 or 2016]
    # bboxes: shape = [ number of anchors x 18], 18 = 4 (bounding box : (cx,cy,w,h) + 14 (7 palm keypoints)

    """Decode palm boxes and seven landmarks using SSD anchors.

    Args:
        score_thresh: Minimum sigmoid confidence for retaining a palm.
        scores: One score logit per anchor.
        bboxes: Per-anchor box and landmark offsets of shape ``(N, 18)``.
        anchors: Normalized center-XY/width/height anchors matching the prediction
            count.
        scale: Model input side length used to normalize the predicted offsets.
        best_only: Compatibility flag for highest-score selection; the standard decoding
            path uses false.

    Returns:
        List of ``HandRegion`` objects with normalized palm boxes and keypoints.
        Negative-width or negative-height boxes are discarded.

    Raises:
        IndexError: If predictions and anchors cannot be selected together. The current
            highest-score path also raises when a candidate meets the threshold.
    """
    regions = []
    scores = 1 / (1 + np.exp(-scores))
    if best_only:
        best_id = np.argmax(scores)
        if scores[best_id] < score_thresh:
            return regions
        det_scores = scores[best_id : best_id + 1]
        det_bboxes2 = bboxes[best_id : best_id + 1]
        det_anchors = anchors[best_id : best_id + 1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0:
            return regions
    try:
        det_bboxes2 = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]
    except Exception as e:
        raise IndexError(
            "Wrong parser scale set. Please use setScale method to set different scale according to model dimensions (e.g. 128)."
        ) from e

    det_bboxes = det_bboxes2 * np.tile(det_anchors[:, 2:4], 9) / scale + np.tile(
        det_anchors[:, 0:2], 9
    )
    det_bboxes[:, 2:4] = det_bboxes[:, 2:4] - det_anchors[:, 0:2]
    det_bboxes[:, 0:2] = det_bboxes[:, 0:2] - det_bboxes[:, 3:4] * 0.5

    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i, 0:4]
        # Decoded detection boxes could have negative values for width/height due
        # to model prediction. Filter out those boxes
        if box[2] < 0 or box[3] < 0:
            continue
        kps = []
        # 0 : wrist
        # 1 : index finger joint
        # 2 : middle finger joint
        # 3 : ring finger joint
        # 4 : little finger joint
        # 5 :
        # 6 : thumb joint
        for kp in range(7):
            kps.append(det_bboxes[i, 4 + kp * 2 : 6 + kp * 2])
        regions.append(HandRegion(float(score), box, kps))
    return regions


def rect_transformation(regions, w, h, no_shift=False):
    """Convert rotated regions to pixel rectangles in place.

    Args:
        regions: Regions already populated by ``detections_to_rect()``.
        w: Source image width in pixels.
        h: Source image height in pixels.
        no_shift: If false, shift toward the fingers and expand the square by 2.9; if
            true, keep the center and use the original longer side.
    """
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    # # Expands and shifts the rectangle that contains the palm so that it's likely
    # # to cover the entire hand.
    # node {
    # calculator: "RectTransformationCalculator"
    # input_stream: "NORM_RECT:raw_roi"
    # input_stream: "IMAGE_SIZE:image_size"
    # output_stream: "roi"
    # options: {
    #     [mediapipe.RectTransformationCalculatorOptions.ext] {
    #     scale_x: 2.6
    #     scale_y: 2.6
    #     shift_y: -0.5
    #     square_long: true
    #     }
    # }
    # IMHO 2.9 is better than 2.6. With 2.6, it may happen that finger tips stay outside of the bouding rotated rectangle
    scale_x = 2.9 if not no_shift else 1
    scale_y = 2.9 if not no_shift else 1
    shift_x = 0
    shift_y = -0.5 if not no_shift else 0
    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = 0
        if rotation == 0:
            region.rect_x_center_a = (region.rect_x_center + width * shift_x) * w
            region.rect_y_center_a = (region.rect_y_center + height * shift_y) * h
        else:
            x_shift = w * width * shift_x * math.cos(
                rotation
            ) - h * height * shift_y * math.sin(rotation)  # / w
            y_shift = w * width * shift_x * math.sin(
                rotation
            ) + h * height * shift_y * math.cos(rotation)  # / h
            region.rect_x_center_a = region.rect_x_center * w + x_shift
            region.rect_y_center_a = region.rect_y_center * h + y_shift

        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * scale_x
        region.rect_h_a = long_side * scale_y
        region.rect_points = rotated_rect_to_points(
            region.rect_x_center_a,
            region.rect_y_center_a,
            region.rect_w_a,
            region.rect_h_a,
            region.rotation,
        )


def rotated_rect_to_points(cx, cy, w, h, rotation):
    """Convert a rotated rectangle into integer pixel corners.

    Args:
        cx: Rectangle center X coordinate.
        cy: Rectangle center Y coordinate.
        w: Rectangle width in pixels.
        h: Rectangle height in pixels.
        rotation: Rotation angle in radians.

    Returns:
        Four integer XY coordinate lists in perimeter order.
    """
    b = math.cos(rotation) * 0.5
    a = math.sin(rotation) * 0.5
    p0x = cx - a * h - b * w
    p0y = cy + b * h - a * w
    p1x = cx + a * h - b * w
    p1y = cy - b * h - a * w
    p2x = int(2 * cx - p0x)
    p2y = int(2 * cy - p0y)
    p3x = int(2 * cx - p1x)
    p3y = int(2 * cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]]


def detections_to_rect(regions):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    # # Converts results of palm detection into a rectangle (normalized by image size)
    # # that encloses the palm and is rotated such that the line connecting center of
    # # the wrist and MCP of the middle finger is aligned with the Y-axis of the
    # # rectangle.
    # node {
    #   calculator: "DetectionsToRectsCalculator"
    #   input_stream: "DETECTION:detection"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "NORM_RECT:raw_roi"
    #   options: {
    #     [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #       rotation_vector_start_keypoint_index: 0  # Center of wrist.
    #       rotation_vector_end_keypoint_index: 2  # MCP of middle finger.
    #       rotation_vector_target_angle_degrees: 90
    #     }
    #   }

    """Add normalized rotated-rectangle geometry to each palm in place.

    Args:
        regions: Hand regions with palm boxes and landmarks. The wrist-to-middle-finger
            direction determines rotation.
    """
    target_angle = math.pi * 0.5  # 90 = pi/2
    for region in regions:
        region.rect_w = region.pd_box[2]
        region.rect_h = region.pd_box[3]
        region.rect_x_center = region.pd_box[0] + region.rect_w / 2
        region.rect_y_center = region.pd_box[1] + region.rect_h / 2

        x0, y0 = region.pd_kps[0]  # wrist center
        x1, y1 = region.pd_kps[2]  # middle finger
        rotation = target_angle - math.atan2(-(y1 - y0), x1 - x0)
        region.rotation = normalize_radians(rotation)


def normalize_radians(angle):
    """Wrap an angle to the interval [-pi, pi).

    Args:
        angle: Input angle in radians.

    Returns:
        Equivalent angle between -pi inclusive and pi exclusive.
    """
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


def decode(bboxes, scores, anchors, threshold=0.5, scale=192):
    """Decode palm predictions and attach rotated pixel rectangles.

    Args:
        bboxes: Per-anchor box and landmark offsets of shape ``(N, 18)``.
        scores: One score logit per anchor.
        anchors: Precomputed anchors matching the model input dimensions.
        threshold: Minimum sigmoid score for keeping a detection.
        scale: Side length of the square model input in pixels.

    Returns:
        List of ``HandRegion`` objects containing normalized palm geometry and
        pixel-space rotated rectangles.
    """
    decoded_bboxes = decode_bboxes(threshold, scores, bboxes, anchors, scale=scale)
    detections_to_rect(decoded_bboxes)
    rect_transformation(decoded_bboxes, scale, scale, no_shift=True)
    return decoded_bboxes


def compute_mediapipe_palm_detections(
    bboxes: np.ndarray,
    scores: np.ndarray,
    *,
    anchors: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    scale: int,
    label_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """Decode MediaPipe palms into rotated detections and apply suppression.

    Args:
        bboxes: Per-anchor palm box and landmark predictions.
        scores: Per-anchor palm score logits.
        anchors: Precomputed anchor coordinates used to decode model predictions.
        conf_threshold: Minimum detection confidence used to filter candidates.
        iou_threshold: Intersection-over-union threshold for non-maximum suppression.
        max_det: Maximum number of detection candidates to retain or consider during
            suppression.
        scale: Side length of the square model input in pixels.
        label_names: Optional class-name lookup indexed by predicted class ID.

    Returns:
        Normalized center-XY/width/height boxes, confidence scores, angles in degrees,
        zero-valued class IDs, and optional class names.
    """
    decoded_bboxes = decode(
        bboxes=bboxes,
        scores=scores,
        anchors=anchors,
        threshold=conf_threshold,
        scale=scale,
    )

    bbox_list = []
    nms_bbox_list = []
    score_list = []
    angle_list = []
    for hand in decoded_bboxes:
        extended_points = np.array(hand.rect_points)

        x_dist = extended_points[3][0] - extended_points[0][0]
        y_dist = extended_points[3][1] - extended_points[0][1]

        angle = np.degrees(np.arctan2(y_dist, x_dist))
        x_center, y_center = np.mean(extended_points, axis=0)
        width = np.linalg.norm(extended_points[0] - extended_points[3])
        height = np.linalg.norm(extended_points[0] - extended_points[1])
        x_min = x_center - width / 2
        y_min = y_center - height / 2

        bbox_list.append([x_center, y_center, width, height])
        nms_bbox_list.append([x_min, y_min, width, height])
        angle_list.append(angle)
        score_list.append(hand.pd_score)

    if len(bbox_list) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([], dtype=int),
            [] if label_names is not None else None,
        )

    indices = cv2.dnn.NMSBoxes(
        nms_bbox_list,
        score_list,
        conf_threshold,
        iou_threshold,
        top_k=max_det,
    )
    indices = np.array(indices).reshape(-1)
    if indices.size == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([], dtype=int),
            [] if label_names is not None else None,
        )

    filtered_bboxes = np.array(bbox_list)[indices].astype(np.float32) / scale
    filtered_scores = np.array(score_list)[indices].astype(np.float32)
    filtered_angles = np.round(np.array(angle_list)[indices], 0)
    filtered_bboxes = np.clip(filtered_bboxes, 0, 1)

    labels = np.zeros(len(filtered_bboxes), dtype=int)
    mapped_label_names = (
        [label_names[label] for label in labels] if label_names is not None else None
    )
    return (
        filtered_bboxes,
        filtered_scores,
        filtered_angles,
        labels,
        mapped_label_names,
    )

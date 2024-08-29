"""mediapipe.py.

Description: This script contains utility functions for decoding the output of the MediaPipe hand tracking model.

This script contains code that is based on or directly taken from a public GitHub repository:
https://github.com/geaxgx/depthai_hand_tracker

Original code author(s): geaxgx

License: MIT License

Copyright (c) [2021] [geax]
"""

import math
from collections import namedtuple

import numpy as np


class HandRegion:
    """
    Attributes:
    pd_score : detection score
    pd_box : detection box [x, y, w, h], normalized [0,1] in the squared image
    pd_kps : detection keypoints coordinates [x, y], normalized [0,1] in the squared image
    rect_x_center, rect_y_center : center coordinates of the rotated bounding rectangle, normalized [0,1] in the squared image
    rect_w, rect_h : width and height of the rotated bounding rectangle, normalized in the squared image (may be > 1)
    rotation : rotation angle of rotated bounding rectangle with y-axis in radian
    rect_x_center_a, rect_y_center_a : center coordinates of the rotated bounding rectangle, in pixels in the squared image
    rect_w, rect_h : width and height of the rotated bounding rectangle, in pixels in the squared image
    rect_points : list of the 4 points coordinates of the rotated bounding rectangle, in pixels
            expressed in the squared image during processing,
            expressed in the source rectangular image when returned to the user
    """

    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
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
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)


def generate_anchors(options):
    """
    option : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
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
        det_bboxes2 = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]

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
    """W, h : image input shape."""
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
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


def generate_anchors_and_decode(bboxes, scores, threshold=0.5, scale=192):
    """Generate anchors and decode bounding boxes for mediapipe hand detection model."""
    anchors = generate_handtracker_anchors(scale, scale)
    decoded_bboxes = decode_bboxes(threshold, scores, bboxes, anchors, scale=scale)
    detections_to_rect(decoded_bboxes)
    rect_transformation(decoded_bboxes, scale, scale, no_shift=True)
    return decoded_bboxes

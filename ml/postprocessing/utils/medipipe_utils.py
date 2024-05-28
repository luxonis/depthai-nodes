import math
import numpy as np
from collections import namedtuple

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
        lm_score: global landmark score
        norm_landmarks : 3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        landmarks : 2D landmark coordinates in pixel in the source rectangular image
        world_landmarks : 3D landmark coordinates in meter
        handedness: float between 0. and 1., > 0.5 for right hand, < 0.5 for left hand,
        label: "left" or "right", handedness translated in a string,
        xyz: real 3D world coordinates of the wrist landmark, or of the palm center (if landmarks are not used),
        xyz_zone: (left, top, right, bottom), pixel coordinates in the source rectangular image 
                of the rectangular zone used to estimate the depth
        gesture: (optional, set in recognize_gesture() when use_gesture==True) string corresponding to recognized gesture ("ONE","TWO","THREE","FOUR","FIVE","FIST","OK","PEACE") 
                or None if no gesture has been recognized
        """
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score # Palm detection score 
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints

    def get_rotated_world_landmarks(self):
        world_landmarks_rotated = self.world_landmarks.copy()
        sin_rot = math.sin(self.rotation)
        cos_rot = math.cos(self.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        world_landmarks_rotated[:,:2] = np.dot(world_landmarks_rotated[:,:2], rot_m)
        return world_landmarks_rotated

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

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
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(math.sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        
        for i,r in enumerate(aspect_ratios):
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
                        new_anchor = [x_center, y_center, anchor_width[anchor_id], anchor_height[anchor_id]]
                        # new_anchor.w = anchor_width[anchor_id]
                        # new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)
        
        layer_id = last_same_stride_layer
    return np.array(anchors)

def generate_handtracker_anchors(input_size_width, input_size_height):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
    anchor_options = SSDAnchorOptions(num_layers=4, 
                            min_scale=0.1484375,
                            max_scale=0.75,
                            input_size_height=input_size_height,
                            input_size_width=input_size_width,
                            anchor_offset_x=0.5,
                            anchor_offset_y=0.5,
                            strides=[8, 16, 16, 16],
                            aspect_ratios= [1.0],
                            reduce_boxes_in_lowest_layer=False,
                            interpolated_scale_aspect_ratio=1.0,
                            fixed_anchor_size=True)
    return generate_anchors(anchor_options)

def decode_bboxes(score_thresh, scores, bboxes, anchors, scale=128, best_only=False):
    """
    wi, hi : NN input shape
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    # Decodes the detection tensors generated by the model, based on
    # the SSD anchors and the specification in the options, into a vector of
    # detections. Each detection describes a detected object.

    https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt :
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 896
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 128.0
            y_scale: 128.0
            h_scale: 128.0
            w_scale: 128.0
            min_score_thresh: 0.5
            }
        }
    }
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 2016
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 192.0
            y_scale: 192.0
            w_scale: 192.0
            h_scale: 192.0
            min_score_thresh: 0.5
            }
        }
    }

    scores: shape = [number of anchors 896 or 2016]
    bboxes: shape = [ number of anchors x 18], 18 = 4 (bounding box : (cx,cy,w,h) + 14 (7 palm keypoints)
    """
    regions = []
    scores = 1 / (1 + np.exp(-scores))
    if best_only:
        best_id = np.argmax(scores)
        if scores[best_id] < score_thresh: return regions
        det_scores = scores[best_id:best_id+1]
        det_bboxes2 = bboxes[best_id:best_id+1]
        det_anchors = anchors[best_id:best_id+1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0: return regions
        det_bboxes2 = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]

    # scale = 128 # x_scale, y_scale, w_scale, h_scale
    # scale = 192 # x_scale, y_scale, w_scale, h_scale

    # cx, cy, w, h = bboxes[i,:4]
    # cx = cx * anchor.w / wi + anchor.x_center 
    # cy = cy * anchor.h / hi + anchor.y_center
    # lx = lx * anchor.w / wi + anchor.x_center 
    # ly = ly * anchor.h / hi + anchor.y_center
    det_bboxes = det_bboxes2* np.tile(det_anchors[:,2:4], 9) / scale + np.tile(det_anchors[:,0:2],9)
    # w = w * anchor.w / wi (in the prvious line, we add anchor.x_center and anchor.y_center to w and h, we need to substract them now)
    # h = h * anchor.h / hi
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    # box = [cx - w*0.5, cy - h*0.5, w, h]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5

    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        # Decoded detection boxes could have negative values for width/height due
        # to model prediction. Filter out those boxes
        if box[2] < 0 or box[3] < 0: continue
        kps = []
        # 0 : wrist
        # 1 : index finger joint
        # 2 : middle finger joint
        # 3 : ring finger joint
        # 4 : little finger joint
        # 5 : 
        # 6 : thumb joint
        # for j, name in enumerate(["0", "1", "2", "3", "4", "5", "6"]):
        #     kps[name] = det_bboxes[i,4+j*2:6+j*2]
        for kp in range(7):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])
        regions.append(HandRegion(float(score), box, kps))
    return regions

def rect_transformation(regions, w, h):
    """
    w, h : image input shape
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
    scale_x = 2.9
    scale_y = 2.9
    shift_x = 0
    shift_y = -0.5
    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = 0
        if rotation == 0:
            region.rect_x_center_a = (region.rect_x_center + width * shift_x) * w
            region.rect_y_center_a = (region.rect_y_center + height * shift_y) * h
        else:
            x_shift = (w * width * shift_x * math.cos(rotation) - h * height * shift_y * math.sin(rotation)) #/ w
            y_shift = (w * width * shift_x * math.sin(rotation) + h * height * shift_y * math.cos(rotation)) #/ h
            region.rect_x_center_a = region.rect_x_center*w + x_shift
            region.rect_y_center_a = region.rect_y_center*h + y_shift

        # square_long: true
        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * scale_x
        region.rect_h_a = long_side * scale_y
        region.rect_points = rotated_rect_to_points(region.rect_x_center_a, region.rect_y_center_a, region.rect_w_a, region.rect_h_a, region.rotation)

def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = math.cos(rotation) * 0.5
    a = math.sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]

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
    
    target_angle = math.pi * 0.5 # 90 = pi/2
    for region in regions:
        
        region.rect_w = region.pd_box[2]
        region.rect_h = region.pd_box[3]
        region.rect_x_center = region.pd_box[0] + region.rect_w / 2
        region.rect_y_center = region.pd_box[1] + region.rect_h / 2

        x0, y0 = region.pd_kps[0] # wrist center
        x1, y1 = region.pd_kps[2] # middle finger
        rotation = target_angle - math.atan2(-(y1 - y0), x1 - x0)
        region.rotation = normalize_radians(rotation)

def normalize_radians(angle):
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))

def non_maxima_suppression(bboxes, iou_threshold):
    if len(bboxes) == 0:
        return []
    
    if bboxes.dtype.kind == 'i':
        bboxes = bboxes.astype('float')

    pick = []

    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0])))

    return bboxes[pick].astype('int')
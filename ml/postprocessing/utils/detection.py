from .medipipe import generate_handtracker_anchors, decode_bboxes, detections_to_rect, rect_transformation

def generate_anchors_and_decode(bboxes, scores, threshold=0.5, scale=192):
    """
    Generate anchors and decode bounding boxes for mediapipe hand detection model.
    """
    anchors = generate_handtracker_anchors(scale, scale)
    decoded_bboxes = decode_bboxes(threshold, scores, bboxes, anchors, scale=scale)
    detections_to_rect(decoded_bboxes)
    rect_transformation(decoded_bboxes, scale, scale)
    return decoded_bboxes
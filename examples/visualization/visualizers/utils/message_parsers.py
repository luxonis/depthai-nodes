import depthai as dai

from depthai_nodes.ml.messages import (
    Classifications,
    Clusters,
    CompositeMessage,
    ImgDetectionsExtended,
    Keypoints,
    Lines,
    Map2D,
    SegmentationMasks,
)


def parse_detection_message(message: dai.ImgDetections):
    """Parses the detection message and returns the detections."""
    detections = message.detections
    return detections


def parse_line_detection_message(message: Lines):
    """Parses the line detection message and returns the lines."""
    lines = message.lines
    return lines


def parse_segmentation_message(message: dai.ImgFrame):
    """Parses the segmentation message and returns the mask."""
    mask = message.getFrame()
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    return mask


def parse_keypoints_message(message: Keypoints):
    """Parses the keypoints message and returns the keypoints."""
    keypoints = message.keypoints
    return keypoints


def parse_classification_message(message: Classifications):
    """Parses the classification message and returns the classification."""
    classes = message.classes
    scores = message.scores
    return classes, scores


def parse_image_message(message: dai.ImgFrame):
    """Parses the image message and returns the image."""
    image = message.getFrame()
    return image


def parser_age_gender_message(message: CompositeMessage):
    """Parses the age-gender message and return the age and scores for all genders."""
    message = message.getData()
    age = message["age"]
    gender = message["gender"]
    gender_scores = gender.scores
    gender_classes = gender.classes

    return age, gender_classes, gender_scores


def parse_yolo_kpts_message(message: ImgDetectionsExtended):
    """Parses the yolo keypoints message and returns the keypoints."""
    detections = message.detections
    return detections


def parse_cluster_message(message: Clusters):
    """Parses the cluster message and returns the clusters."""
    clusters = message.clusters
    return clusters


def parse_fast_sam_message(message: SegmentationMasks):
    """Parses the fast sam message and returns the masks."""
    masks = message.masks
    return masks


def parse_map_message(message: Map2D):
    """Parses the map message and returns the map."""
    map = message.map
    return map

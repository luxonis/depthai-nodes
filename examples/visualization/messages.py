import depthai as dai

from depthai_nodes.ml.messages import (
    AgeGender,
    Classifications,
    ImgDetectionsExtended,
    Keypoints,
    Lines,
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


def parser_age_gender_message(message: AgeGender):
    """Parses the age-gender message and return the age and scores for all genders."""

    age = message.age
    gender = message.gender
    gender_scores = gender.scores
    gender_classes = gender.classes

    return age, gender_classes, gender_scores


def parse_yolo_kpts_message(message: ImgDetectionsExtended):
    """Parses the yolo keypoints message and returns the keypoints."""
    detections = message.detections
    return detections

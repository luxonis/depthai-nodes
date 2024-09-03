from .classification import visualize_age_gender, visualize_classification
from .detection import (
    visualize_detections,
    visualize_line_detections,
    visualize_yolo_extended,
)
from .image import visualize_image
from .keypoints import visualize_keypoints
from .segmentation import visualize_segmentation

parser_mapping = {
    "YuNetParser": visualize_detections,
    "SCRFDParser": visualize_detections,
    "MPPalmDetectionParser": visualize_detections,
    "YOLO": visualize_detections,
    "SSD": visualize_detections,
    "SegmentationParser": visualize_segmentation,
    "MLSDParser": visualize_line_detections,
    "KeypointParser": visualize_keypoints,
    "HRNetParser": visualize_keypoints,
    "SuperAnimalParser": visualize_keypoints,
    "MPHandLandmarkParser": visualize_keypoints,
    "ClassificationParser": visualize_classification,
    "ImageOutputParser": visualize_image,
    "MonocularDepthParser": visualize_image,
    "AgeGenderParser": visualize_age_gender,
    "YOLOExtendedParser": visualize_yolo_extended,
}

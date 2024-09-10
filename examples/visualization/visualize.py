import depthai as dai
from visualizers import (
    visualize_age_gender,
    visualize_classification,
    visualize_detections,
    visualize_fastsam,
    visualize_image,
    visualize_keypoints,
    visualize_line_detections,
    visualize_map,
    visualize_segmentation,
    visualize_yolo_extended,
)

visualizers = {
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
    "MapOutputParser": visualize_map,
    "AgeGenderParser": visualize_age_gender,
    "YOLOExtendedParser": visualize_yolo_extended,
    "FastSAMParser": visualize_fastsam,
}


def visualize(
    frame: dai.ImgFrame, message: dai.Buffer, parser_name: str, extraParams: dict
):
    """Calls the appropriate visualizer based on the parser name and returns True if the
    pipeline should be stopped."""
    visualizer = visualizers[parser_name]
    return visualizer(frame, message, extraParams)

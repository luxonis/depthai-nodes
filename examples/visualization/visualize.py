import depthai as dai

from .visualizers import (
    visualize_age_gender,
    visualize_classification,
    visualize_detections,
    visualize_fastsam,
    visualize_image,
    visualize_keypoints,
    visualize_lane_detections,
    visualize_line_detections,
    visualize_map,
    visualize_multi_classification,
    visualize_segmentation,
    visualize_text_detection,
    visualize_text_recognition,
    visualize_yolo_extended,
)

visualizers_dict = {
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
    "LaneDetectionParser": visualize_lane_detections,
    "FastSAMParser": visualize_fastsam,
    "PPTextDetectionParser": visualize_text_detection,
    "PaddleOCRParser": visualize_text_recognition,
    "MultiClassificationParser": visualize_multi_classification,
}


def visualize(
    frame: dai.ImgFrame, message: dai.Buffer, parser_name: str, extraParams: dict
):
    """Calls the appropriate visualizer based on the parser name and returns True if the
    pipeline should be stopped."""
    visualizer = visualizers_dict[parser_name]
    return visualizer(frame, message, extraParams)

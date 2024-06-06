import depthai as dai

from ....messages.img_detections import (
    ImgDetectionWithKeypoints,
    ImgDetectionsWithKeypoints,
)


def create_detections_msg(
    detections: list,
    include_keypoints: bool = False,
) -> dai.ImgFrame:
    """
    Create a depthai message for an image array.

    @type detections: list
    @ivar detections: List of detections.

    @type include_keypoints: bool
    @ivar include_keypoints: If True, the keypoints are included in the message.
    """

    if include_keypoints:
        img_detection = ImgDetectionWithKeypoints
        img_detections = ImgDetectionsWithKeypoints
    else:
        img_detection = dai.ImgDetection
        img_detections = dai.ImgDetections

    img_detection_list = []
    for detection in detections:
        detections_message = img_detection()
        detections_message.label = detection["label"]
        detections_message.confidence = detection["score"]
        detections_message.xmin = detection["bbox"][0]
        detections_message.ymin = detection["bbox"][1]
        detections_message.xmax = detection["bbox"][0] + detection["bbox"][2]
        detections_message.ymax = detection["bbox"][1] + detection["bbox"][3]
        if include_keypoints:
            detections_message.keypoints = detection["keypoints"]
        img_detection_list.append(detections_message)

    detections_message = img_detections()
    detections_message.detections = img_detection_list

    return detections_message

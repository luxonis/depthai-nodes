from typing import Union

import depthai as dai

from depthai_nodes.message.img_detections import ImgDetectionExtended


def compute_area(
    detection: Union[dai.ImgDetection, ImgDetectionExtended, dai.SpatialImgDetection],
):
    """Computes the normalized area of a detection bounding box.

    @param detection: Detection object to compute the area for. Can be of type
        dai.ImgDetection, ImgDetectionExtended, or dai.SpatialImgDetection.
    @type detection: Union[dai.ImgDetection, ImgDetectionExtended,
        dai.SpatialImgDetection]
    @return: Normalized area (width * height) of the detection bounding box.
    @rtype: float
    """

    if isinstance(detection, ImgDetectionExtended):
        width = detection.rotated_rect.size.width
        height = detection.rotated_rect.size.height
        area = width * height
    elif isinstance(detection, (dai.ImgDetection, dai.SpatialImgDetection)):
        width = detection.xmax - detection.xmin
        height = detection.ymax - detection.ymin
        area = width * height
    else:
        raise TypeError(f"Unsupported detection type: {type(detection).__name__}")

    return area

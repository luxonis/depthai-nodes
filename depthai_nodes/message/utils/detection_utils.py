from typing import Union

import depthai as dai


def compute_area(
    detection: Union[dai.ImgDetection, dai.SpatialImgDetection],
):
    """Computes the normalized area of a detection bounding box.

    @param detection: Detection object to compute the area for. Can be of type
        dai.ImgDetection, or dai.SpatialImgDetection.
    @type detection: Union[dai.ImgDetection, dai.SpatialImgDetection]
    @return: Normalized area (width * height) of the detection bounding box.
    @rtype: float
    """
    if isinstance(detection, (dai.ImgDetection, dai.SpatialImgDetection)):
        width = detection.xmax - detection.xmin
        height = detection.ymax - detection.ymin
        area = width * height
    else:
        raise TypeError(f"Unsupported detection type: {type(detection).__name__}")

    return area

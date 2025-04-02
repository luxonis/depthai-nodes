import copy
from typing import Union

import depthai as dai


def copy_message(msg: dai.Buffer) -> dai.Buffer:
    """Copies the incoming message and returns it.

    @param msg: The input message.
    @type msg: dai.Buffer
    @return: The copied message.
    @rtype: dai.Buffer
    """

    # 1st attempt: native .copy() method
    if hasattr(msg, "copy"):
        return msg.copy()

    # 2nd attempt: custom copy implementation
    try:
        return _copy(msg)
    except TypeError:
        pass

    # 3rd attempt: deepcopy (the most general approach)
    try:
        return copy.deepcopy(msg)
    except TypeError as e:
        raise TypeError(f"Copying of message type {type(msg)} is not supported.") from e


def _copy(msg: dai.Buffer) -> dai.Buffer:
    def _copy_img_detection(
        img_det: Union[dai.ImgDetection, dai.SpatialImgDetection],
    ) -> Union[dai.ImgDetection, dai.SpatialImgDetection]:
        assert isinstance(img_det, (dai.ImgDetection, dai.SpatialImgDetection))
        if isinstance(img_det, dai.ImgDetection):
            img_det_copy = dai.ImgDetection()
        if isinstance(img_det, dai.SpatialImgDetection):
            img_det_copy = dai.SpatialImgDetection()
            img_det_copy.spatialCoordinates = img_det.spatialCoordinates
            img_det_copy.boundingBoxMapping = img_det.boundingBoxMapping
        img_det_copy.xmin = img_det.xmin
        img_det_copy.ymin = img_det.ymin
        img_det_copy.xmax = img_det.xmax
        img_det_copy.ymax = img_det.ymax
        img_det_copy.label = img_det.label
        img_det_copy.confidence = img_det.confidence
        return img_det_copy

    def _copy_img_detections(
        img_dets: Union[dai.ImgDetections, dai.SpatialImgDetections],
    ) -> Union[dai.ImgDetections, dai.SpatialImgDetections]:
        assert isinstance(img_dets, (dai.ImgDetections, dai.SpatialImgDetections))
        if isinstance(img_dets, dai.ImgDetections):
            img_dets_copy = dai.ImgDetections()
        if isinstance(img_dets, dai.SpatialImgDetections):
            img_dets_copy = dai.SpatialImgDetections()
        img_dets_copy.detections = [
            _copy_img_detection(img_det) for img_det in img_dets.detections
        ]
        img_dets_copy.setSequenceNum(img_dets.getSequenceNum())
        img_dets_copy.setTimestamp(img_dets.getTimestamp())
        img_dets_copy.setTimestampDevice(img_dets.getTimestampDevice())
        img_dets_copy.setTransformation(img_dets.getTransformation())
        return img_dets_copy

    if isinstance(msg, (dai.ImgDetection, dai.SpatialImgDetection)):
        return _copy_img_detection(msg)
    elif isinstance(msg, (dai.ImgDetections, dai.SpatialImgDetections)):
        return _copy_img_detections(msg)
    else:
        # TODO: define logic for copying other message types
        raise TypeError(f"Copying of message type {type(msg)} is not supported.")

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
    def _copy_metadata(msg: dai.Buffer) -> dai.Buffer:
        msg_type = type(msg)
        msg_copy = msg_type()
        if hasattr(msg, "getSequenceNum"):
            msg_copy.setSequenceNum(msg.getSequenceNum())
        if hasattr(msg, "getTimestamp"):
            msg_copy.setTimestamp(msg.getTimestamp())
        if hasattr(msg, "getTimestampDevice"):
            msg_copy.setTimestampDevice(msg.getTimestampDevice())
        if hasattr(msg, "getTransformation"):
            msg_copy.setTransformation(msg.getTransformation())
        return msg_copy

    def _copy_img_frame(img_frame: dai.ImgFrame) -> dai.ImgFrame:
        img_frame_copy = _copy_metadata(img_frame)
        img_frame_copy.setCvFrame(img_frame.getCvFrame(), img_frame.getType())
        img_frame_copy.setCategory(img_frame.getCategory())
        return img_frame_copy

    def _copy_img_detection(
        img_det: Union[dai.ImgDetection, dai.SpatialImgDetection],
    ) -> Union[dai.ImgDetection, dai.SpatialImgDetection]:
        assert isinstance(img_det, (dai.ImgDetection, dai.SpatialImgDetection))
        img_det_copy = _copy_metadata(img_det)
        if isinstance(img_det, dai.SpatialImgDetection):
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
        img_dets_copy = _copy_metadata(img_dets)
        img_dets_copy.detections = [
            _copy_img_detection(img_det) for img_det in img_dets.detections
        ]
        return img_dets_copy

    def _copy_point2f(point2f: dai.Point2f) -> dai.Point2f:
        point2f_copy = _copy_metadata(point2f)
        point2f_copy.x = point2f.x
        point2f_copy.y = point2f.y
        # TODO: set the value for .isNormalized()
        return point2f_copy

    if isinstance(msg, dai.ImgFrame):
        return _copy_img_frame(msg)
    elif isinstance(msg, (dai.ImgDetection, dai.SpatialImgDetection)):
        return _copy_img_detection(msg)
    elif isinstance(msg, (dai.ImgDetections, dai.SpatialImgDetections)):
        return _copy_img_detections(msg)
    elif isinstance(msg, dai.Point2f):
        return _copy_point2f(msg)
    else:
        # TODO: define logic for copying other message types
        raise TypeError(f"Copying of message type {type(msg)} is not supported.")

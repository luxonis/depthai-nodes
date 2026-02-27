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
        assert isinstance(img_det_copy, (dai.ImgDetection, dai.SpatialImgDetection))
        if isinstance(img_det, dai.SpatialImgDetection):
            img_det_copy.spatialCoordinates = img_det.spatialCoordinates
            img_det_copy.boundingBoxMapping = img_det.boundingBoxMapping
        img_det_copy.label = img_det.label
        img_det_copy.labelName = img_det.labelName
        img_det_copy.confidence = img_det.confidence
        img_det_copy.setBoundingBox(_copy_rotated_rect(img_det.getBoundingBox()))
        img_det_copy.setKeypoints(_copy_keypoints(img_det.getKeypoints()))
        img_det_copy.setEdges(copy.deepcopy(img_det.getEdges()))
        return img_det_copy

    def _copy_img_detections(
        img_dets: Union[dai.ImgDetections, dai.SpatialImgDetections],
    ) -> Union[dai.ImgDetections, dai.SpatialImgDetections]:
        assert isinstance(img_dets, (dai.ImgDetections, dai.SpatialImgDetections))
        img_dets_copy = _copy_metadata(img_dets)
        if isinstance(img_dets, dai.ImgDetections):
            assert isinstance(img_dets_copy, dai.ImgDetections)
            masks = img_dets.getCvSegmentationMask()
            if masks is not None:
                img_dets_copy.setCvSegmentationMask(masks)
        img_dets_copy.detections = [
            _copy_img_detection(img_det) for img_det in img_dets.detections
        ]
        return img_dets_copy

    def _copy_keypoints(keypoints: list[dai.Keypoint]) -> list[dai.Keypoint]:
        keypoints_copy = [_copy_keypoint(keypoint) for keypoint in keypoints]
        return keypoints_copy

    def _copy_point2f(point2f: dai.Point2f) -> dai.Point2f:
        point2f_copy = _copy_metadata(point2f)
        point2f_copy.x = point2f.x
        point2f_copy.y = point2f.y
        return point2f_copy

    def _copy_size2f(size2f: dai.Size2f) -> dai.Size2f:
        size2f_copy = _copy_metadata(size2f)
        size2f_copy.width = size2f.width
        size2f_copy.height = size2f.height
        return size2f_copy

    def _copy_point3f(point3f: dai.Point3f) -> dai.Point3f:
        point3f_copy = _copy_metadata(point3f)
        point3f_copy.x = point3f.x
        point3f_copy.y = point3f.y
        point3f_copy.z = point3f.z
        return point3f_copy

    def _copy_keypoint(keypoint: dai.Keypoint) -> dai.Keypoint:
        keypoint_copy = _copy_metadata(keypoint)
        keypoint_copy.imageCoordinates = _copy_point3f(keypoint.imageCoordinates)
        keypoint_copy.confidence = keypoint.confidence
        keypoint_copy.labelName = keypoint.labelName
        return keypoint_copy

    def _copy_rotated_rect(rotated_rect: dai.RotatedRect) -> dai.RotatedRect:
        rotated_rect_copy: dai.RotatedRect = _copy_metadata(rotated_rect)
        rotated_rect_copy.center = _copy_point2f(rotated_rect.center)
        rotated_rect_copy.size = _copy_size2f(rotated_rect.size)
        rotated_rect_copy.angle = rotated_rect.angle
        return rotated_rect_copy

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

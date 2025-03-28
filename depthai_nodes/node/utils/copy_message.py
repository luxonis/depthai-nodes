import depthai as dai

import copy


def copy_message(msg: dai.Buffer) -> dai.Buffer:
    """Copies the incoming message and returns it.

    @param msg: The input message.
    @type msg: dai.Buffer
    @return: The copied message.
    @rtype: dai.Buffer
    """

    try:
        return copy.deepcopy(msg)  # First attempt: use deepcopy
    except Exception:
        pass  # If deepcopy fails, move to the next approach

    try:
        return msg.copy()  # Second attempt: use .copy() method
    except AttributeError:
        pass  # If .copy() method doesn't exist, move to the next approach

    return _copy(msg)  # Last attempt: use a custom copy implementation


def _copy(msg: dai.Buffer) -> dai.Buffer:

    def _copy_img_detection(img_det: dai.ImgDetection) -> dai.ImgDetection:
        assert isinstance(img_det, dai.ImgDetection)
        img_det_copy = dai.ImgDetection()
        img_det_copy.xmin = img_det.xmin
        img_det_copy.ymin = img_det.ymin
        img_det_copy.xmax = img_det.xmax
        img_det_copy.ymax = img_det.ymax
        img_det_copy.label = img_det.label
        img_det_copy.confidence = img_det.confidence
        return img_det_copy

    def _copy_img_detections(img_dets: dai.ImgDetections) -> dai.ImgDetections:
        assert isinstance(img_dets, dai.ImgDetections)
        img_dets_copy = dai.ImgDetections()
        img_dets_copy.detections = [
            _copy_img_detection(img_det) for img_det in img_dets.detections
        ]
        img_dets_copy.setSequenceNum(img_dets.getSequenceNum())
        img_dets_copy.setTimestamp(img_dets.getTimestamp())
        img_dets_copy.setTimestampDevice(img_dets.getTimestampDevice())
        img_dets_copy.setTransformation(img_dets.getTransformation())
        return img_dets_copy

    if isinstance(msg, dai.ImgDetection):
        return _copy_img_detection(msg)
    elif isinstance(msg, dai.ImgDetections):
        return _copy_img_detections(msg)
    else:
        # TODO: define logic for copying other message types
        raise TypeError(f"Copying of message type {type(msg)} is not supported.")

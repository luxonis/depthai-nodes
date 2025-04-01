import copy

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
    except Exception:
        raise TypeError(f"Copying of message type {type(msg)} is not supported.")


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

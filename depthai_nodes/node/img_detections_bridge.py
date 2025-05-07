from typing import Dict

import depthai as dai

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.logging import get_logger
from depthai_nodes.node.base_host_node import BaseHostNode


class ImgDetectionsBridge(BaseHostNode):
    """Transforms the dai.ImgDetections to ImgDetectionsExtended object or vice versa.
    Note that conversion from ImgDetectionsExtended to ImgDetection loses information
    about segmentation, keypoints and rotation.

    Attributes
    ----------
    input : dai.ImgDetections or ImgDetectionsExtended
        The input message for the ImgDetections object.
    output : dai.ImgDetections or ImgDetectionsExtended
        The output message of the transformed ImgDetections object.
    """

    def __init__(self) -> None:
        super().__init__()
        self._logger = get_logger()
        self._log = True
        self._ignore_angle = False
        self._label_encoding = {}

    def setIgnoreAngle(self, ignore_angle: bool) -> bool:
        """Sets whether to ignore the angle of the detections during transformation.

        @param ignore_angle: Whether to ignore the angle of the detections.
        @type ignore_angle: bool
        """
        if not isinstance(ignore_angle, bool):
            raise ValueError("ignore_angle must be a boolean.")
        self._ignore_angle = ignore_angle

    def setLabelEncoding(self, label_encoding: Dict[int, str]) -> None:
        """Sets the label encoding.

        @param label_encoding: The label encoding with labels as keys and label names as
            values.
        @type label_encoding: Dict[int, str]
        """
        if not isinstance(label_encoding, Dict):
            raise ValueError("label_encoding must be a dictionary.")
        self._label_encoding = label_encoding

    def build(
        self,
        msg: dai.Node.Output,
        ignore_angle: bool = False,
        label_encoding: Dict[int, str] = {},
    ) -> "ImgDetectionsBridge":
        """Configures the node connections.

        @param msg: The input message for the ImgDetections object.
        @type msg: dai.Node.Output
        @param ignore_angle: Whether to ignore the angle of the detections.
        @type ignore_angle: bool
        @param label_encoding: The label encoding with labels as keys and label names as
            values.
        @type label_encoding: Dict[int, str]
        @return: The node object with the transformed ImgDetections object.
        @rtype: ImgDetectionsBridge
        """
        self.link_args(msg)
        self.setIgnoreAngle(ignore_angle)
        self.setLabelEncoding(label_encoding)
        return self

    def process(self, msg: dai.Buffer) -> None:
        """Transforms the incoming ImgDetections object.

        @param msg: The input message for the ImgDetections object.
        @type msg: dai.ImgDetections or ImgDetectionsExtended
        """

        if isinstance(msg, dai.ImgDetections):
            msg_transformed = self._img_det_to_img_det_ext(msg)
        elif isinstance(msg, ImgDetectionsExtended):
            msg_transformed = self._img_det_ext_to_img_det(msg)
            if self._log:
                self._logger.warning(
                    "You are using ImgDetectionsBridge to transform from ImgDetectionsExtended to ImgDetections. This results in lose of keypoint, segmentation and bbox rotation information if present in the original message."
                )
                self._log = False  # only log once
        else:
            raise TypeError(
                f"Expected dai.ImgDetections or ImgDetectionsExtended, got {type(msg)}"
            )

        msg_transformed.setTimestamp(msg.getTimestamp())
        msg_transformed.setSequenceNum(msg.getSequenceNum())
        msg_transformed.setTransformation(msg.getTransformation())
        self.out.send(msg_transformed)

    def _img_det_to_img_det_ext(
        self, img_dets: dai.ImgDetections
    ) -> ImgDetectionsExtended:
        """Transforms the incoming dai.ImgDetections object to
        ImgDetectionsExtended."."""
        assert isinstance(img_dets, dai.ImgDetections)

        img_dets_ext = ImgDetectionsExtended()

        detections_transformed = []
        for detection in img_dets.detections:
            detection_transformed = ImgDetectionExtended()
            detection_transformed.label = detection.label
            label_name = self._label_encoding.get(detection.label)
            if label_name is not None:
                detection_transformed.label_name = label_name
            detection_transformed.confidence = detection.confidence
            x_center = (detection.xmin + detection.xmax) / 2
            y_center = (detection.ymin + detection.ymax) / 2
            width = detection.xmax - detection.xmin
            height = detection.ymax - detection.ymin
            detection_transformed.rotated_rect = (
                x_center,
                y_center,
                width,
                height,
                0,  # dai.ImgDetections has no angle info
            )
            detections_transformed.append(detection_transformed)

        img_dets_ext.detections = detections_transformed

        return img_dets_ext

    def _img_det_ext_to_img_det(
        self, img_det_ext: ImgDetectionsExtended
    ) -> dai.ImgDetections:
        """Transforms the incoming ImgDetectionsExtended object to
        dai.ImgDetections."."""
        assert isinstance(img_det_ext, ImgDetectionsExtended)

        img_dets = dai.ImgDetections()

        detections_transformed = []
        for detection in img_det_ext.detections:
            detection_transformed = dai.ImgDetection()
            detection_transformed.label = detection.label
            detection_transformed.confidence = detection.confidence
            if not self._ignore_angle and detection.rotated_rect.angle != 0:
                raise NotImplementedError(
                    "Unable to transform ImgDetectionsExtended with rotation."
                )
            xmin, ymin, xmax, ymax = detection.rotated_rect.getOuterRect()
            detection_transformed.xmin = xmin
            detection_transformed.ymin = ymin
            detection_transformed.xmax = xmax
            detection_transformed.ymax = ymax

            detections_transformed.append(detection_transformed)

        img_dets.detections = detections_transformed

        return img_dets

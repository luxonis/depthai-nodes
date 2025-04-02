import depthai as dai

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended


class ImgDetectionsBridge(dai.node.HostNode):
    """Transforms the dai.ImgDetections to ImgDetectionsExtended object or vice versa".

    Attributes
    ----------
    input : dai.ImgDetections or ImgDetectionsExtended
        The input message for the ImgDetections object.
    output : dai.ImgDetections or ImgDetectionsExtended
        The output message of the transformed ImgDetections object.
    """

    def __init__(self) -> None:
        super().__init__()

    def setIgnoreAngle(self, ignore_angle: bool) -> bool:
        """Sets whether to ignore the angle of the detections during transformation.

        @param ignore_angle: Whether to ignore the angle of the detections.
        @type ignore_angle: bool
        """
        if not isinstance(ignore_angle, bool):
            raise ValueError("ignore_angle must be a boolean.")
        self._ignore_angle = ignore_angle
        return self._ignore_angle

    def build(
        self, msg: dai.Node.Output, ignore_angle: bool = False
    ) -> "ImgDetectionsBridge":
        """Configures the node connections.

        @param msg: The input message for the ImgDetections object.
        @type msg: dai.Node.Output
        @param ignore_angle: Whether to ignore the angle of the detections.
        @type ignore_angle: bool
        @return: The node object with the transformed ImgDetections object.
        @rtype: ImgDetectionsBridge
        """
        self.link_args(msg)
        self.ignore_angle = self.setIgnoreAngle(ignore_angle)
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
            if not self.ignore_angle and detection.rotated_rect.angle != 0:
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

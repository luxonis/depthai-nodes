import depthai as dai
import numpy as np

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.logging import get_logger
from depthai_nodes.message.creators.keypoints import create_keypoints_message
from depthai_nodes.node.base_host_node import BaseHostNode


class ImgDetectionsBridge(BaseHostNode):
    """Transforms the dai.ImgDetections to ImgDetectionsExtended object or vice versa.

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
        self._logger.debug("ImgDetectionsBridge initialized")

    def build(
        self,
        msg: dai.Node.Output,
    ) -> "ImgDetectionsBridge":
        """Configures the node connections.

        @param msg: The input message for the ImgDetections object.
        @type msg: dai.Node.Output
        @return: The node object with the transformed ImgDetections object.
        @rtype: ImgDetectionsBridge
        """
        self.link_args(msg)
        self._logger.debug("ImgDetectionsBridge built.")
        return self

    def process(self, msg: dai.Buffer) -> None:
        """Transforms the incoming ImgDetections object.

        @param msg: The input message for the ImgDetections object.
        @type msg: dai.ImgDetections or ImgDetectionsExtended
        """
        self._logger.debug("Processing new input")
        if isinstance(msg, dai.ImgDetections):
            msg_transformed = self._img_det_to_img_det_ext(msg)
        elif isinstance(msg, ImgDetectionsExtended):
            msg_transformed = self._img_det_ext_to_img_det(msg)
            if self._log:
                self._logger.warning(
                    "You are using ImgDetectionsBridge to transform from ImgDetectionsExtended to ImgDetections."
                )
                self._log = False  # only log once
        else:
            raise TypeError(
                f"Expected dai.ImgDetections or ImgDetectionsExtended, got {type(msg)}"
            )

        msg_transformed.setTimestamp(msg.getTimestamp())
        msg_transformed.setSequenceNum(msg.getSequenceNum())
        msg_transformed.setTimestampDevice(msg.getTimestampDevice())
        transformation = msg.getTransformation()
        if transformation is not None:
            msg_transformed.setTransformation(transformation)

        self._logger.debug("Detection message created")

        self.out.send(msg_transformed)

        self._logger.debug("Message sent successfully")

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
            detection_transformed.label_name = detection.labelName
            detection_transformed.confidence = detection.confidence
            x_center = detection.getBoundingBox().center.x
            y_center = detection.getBoundingBox().center.y
            width = detection.getBoundingBox().size.width
            height = detection.getBoundingBox().size.height
            angle = detection.getBoundingBox().angle
            detection_transformed.rotated_rect = (
                x_center,
                y_center,
                width,
                height,
                angle,
            )
            kpts = detection.getKeypoints()
            if kpts is not None:
                kpts_list = [
                    [
                        kp.imageCoordinates.x,
                        kp.imageCoordinates.y,
                        kp.imageCoordinates.z,
                    ]
                    for kp in kpts
                ]
                scores_list = [kp.confidence for kp in kpts]
                if all(score == -1 for score in scores_list):
                    scores_list = None
                edges_list = [(edge[0], edge[1]) for edge in detection.getEdges()]
                keypoint_label_names_list = [kp.labelName for kp in kpts]
                if any(label_name is None for label_name in keypoint_label_names_list):
                    keypoint_label_names_list = None
                keypoints_msg = create_keypoints_message(
                    keypoints=kpts_list,
                    scores=scores_list,
                    edges=edges_list,
                    label_names=keypoint_label_names_list,
                )
                detection_transformed.keypoints = keypoints_msg
            detections_transformed.append(detection_transformed)

        mask = img_dets.getCvSegmentationMask()
        if mask is not None:
            mask = np.astype(mask, np.int16)
            mask[mask == 255] = -1
            img_dets_ext.masks = mask

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
            detection_transformed.label = (
                0 if detection.label == -1 else detection.label
            )
            detection_transformed.labelName = detection.label_name
            detection_transformed.confidence = detection.confidence
            detection_transformed.setBoundingBox(detection.rotated_rect)
            edges = detection.edges
            edges = [[edge[0], edge[1]] for edge in edges]
            kpts = detection.keypoints
            kpts_list = [[kp.x, kp.y, kp.z] for kp in kpts]
            scores_list = [kp.confidence for kp in kpts]
            kpts_transformed = [
                dai.Keypoint(x=kp[0], y=kp[1], z=kp[2], confidence=score)
                for kp, score in zip(kpts_list, scores_list)
            ]
            detection_transformed.setKeypoints(kpts_transformed, edges)

            detections_transformed.append(detection_transformed)

        img_dets.detections = detections_transformed

        mask = img_det_ext.masks
        if mask is not None:
            mask[
                mask == -1
            ] = 255  # replace -1 with 255 to match dai.ImgDetections BG class
            img_dets.setCvSegmentationMask(mask.astype(np.uint8))

        return img_dets

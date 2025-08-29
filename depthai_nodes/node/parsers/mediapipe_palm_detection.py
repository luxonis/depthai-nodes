from typing import Any, Dict, List

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.detection import DetectionParser
from depthai_nodes.node.parsers.utils.medipipe import (
    decode,
    generate_handtracker_anchors,
)


class MPPalmDetectionParser(DetectionParser):
    """Parser class for parsing the output of the Mediapipe Palm detection model. As the
    result, the node sends out the detected hands in the form of a message containing
    bounding boxes, labels, and confidence scores.

    Attributes
    ----------
    output_layer_names: List[str]
        Names of the output layers relevant to the parser.
    conf_threshold : float
        Confidence score threshold for detected hands.
    iou_threshold : float
        Non-maximum suppression threshold.
    max_det : int
        Maximum number of detections to keep.
    scale : int
        Scale of the input image.

    Output Message/s
    -------
    **Type**: ImgDetectionsExtended

    **Description**: ImgDetectionsExtended message containing bounding boxes, labels, and confidence scores of detected hands.

    See also
    --------
    Official MediaPipe Hands solution:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    """

    def __init__(
        self,
        output_layer_names: List[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_det: int = 100,
        scale: int = 192,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_names: Names of the output layers relevant to the parser.
        @type output_layer_names: List[str]
        @param conf_threshold: Confidence score threshold for detected hands.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        @param scale: Scale of the input image.
        @type scale: int
        """
        super().__init__(conf_threshold, iou_threshold, max_det)
        self.output_layer_names = (
            [] if output_layer_names is None else output_layer_names
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.scale = scale
        self.label_names = ["Palm"]
        self._anchors = generate_handtracker_anchors(scale, scale)
        self._logger.debug(
            f"MPPalmDetectionParser initialized with output_layer_names={output_layer_names}, conf_threshold={conf_threshold}, iou_threshold={iou_threshold}, max_det={max_det}, scale={scale}"
        )

    def setOutputLayerNames(self, output_layer_names: List[str]) -> None:
        """Sets the output layer name(s) for the parser.

        @param output_layer_names: The name of the output layer(s) from which the scores
            are extracted.
        @type output_layer_names: List[str]
        """
        if not isinstance(output_layer_names, list):
            raise ValueError("Output layer name must be a list.")
        if not all(isinstance(layer_name, str) for layer_name in output_layer_names):
            raise ValueError("Each output layer name must be a string.")
        if len(output_layer_names) != 2:
            raise ValueError(
                f"Only two output layers are supported for MPPalmDetectionParser, got {len(output_layer_names)} layers."
            )
        self.output_layer_names = output_layer_names
        self._logger.debug(f"Output layer names set to {self.output_layer_names}")

    def setScale(self, scale: int) -> None:
        """Sets the scale of the input image.

        @param scale: Scale of the input image.
        @type scale: int
        """
        if not isinstance(scale, int):
            raise ValueError("Scale must be an integer.")
        self.scale = scale
        self._logger.debug(f"Scale set to {self.scale}")

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "MPPalmDetectionParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: MPPalmDetectionParser
        """

        super().build(head_config)
        output_layers = head_config.get("outputs", [])
        if len(output_layers) != 2:
            raise ValueError(
                f"Only two output layers are supported for MPPalmDetectionParser, got {len(output_layers)} layers."
            )
        self.output_layer_names = output_layers
        self.scale = head_config.get("scale", self.scale)
        self._anchors = generate_handtracker_anchors(self.scale, self.scale)

        self._logger.debug(
            f"MPPalmDetectionParser built with output_layer_names={self.output_layer_names}, scale={self.scale}"
        )

        return self

    def run(self):
        self._logger.debug("MPPalmDetectionParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            all_tensors = output.getAllLayerNames()

            self._logger.debug(f"Processing input with layers: {all_tensors}")

            bboxes = None
            scores = None

            for tensor_name in all_tensors:
                tensor = np.array(
                    output.getTensor(tensor_name, dequantize=True), dtype=np.float32
                )

                if bboxes is None:
                    bboxes = tensor
                    scores = tensor
                else:
                    bboxes = bboxes if tensor.shape[-1] < bboxes.shape[-1] else tensor
                    scores = tensor if tensor.shape[-1] < scores.shape[-1] else scores

            bboxes = bboxes.reshape(-1, 18)
            scores = scores.reshape(-1)

            if bboxes is None or scores is None:
                raise ValueError("No valid output tensors found.")

            decoded_bboxes = decode(
                bboxes=bboxes,
                scores=scores,
                anchors=self._anchors,
                threshold=self.conf_threshold,
                scale=self.scale,
            )

            bboxes = []
            scores = []
            angles = []
            for hand in decoded_bboxes:
                extended_points = np.array(hand.rect_points)

                x_dist = extended_points[3][0] - extended_points[0][0]
                y_dist = extended_points[3][1] - extended_points[0][1]

                angle = np.degrees(np.arctan2(y_dist, x_dist))
                x_center, y_center = np.mean(extended_points, axis=0)
                width = np.linalg.norm(extended_points[0] - extended_points[3])
                height = np.linalg.norm(extended_points[0] - extended_points[1])

                bboxes.append([x_center, y_center, width, height])
                angles.append(angle)
                scores.append(hand.pd_score)

            indices = cv2.dnn.NMSBoxes(
                bboxes,
                scores,
                self.conf_threshold,
                self.iou_threshold,
                top_k=self.max_det,
            )
            bboxes = np.array(bboxes)[indices]
            scores = np.array(scores)[indices]
            angles = np.array(angles)[indices]
            bboxes = bboxes.astype(float) / self.scale

            bboxes = np.clip(bboxes, 0, 1)
            angles = np.round(angles, 0)

            labels = np.array([0] * len(bboxes))

            label_names = (
                [self.label_names[label] for label in labels]
                if self.label_names
                else None
            )
            detections_msg = create_detection_message(
                bboxes=bboxes,
                scores=scores,
                angles=angles,
                labels=labels,
                label_names=label_names,
            )
            detections_msg.setTimestamp(output.getTimestamp())
            detections_msg.setSequenceNum(output.getSequenceNum())
            detections_msg.setTimestampDevice(output.getTimestampDevice())
            transformation = output.getTransformation()
            if transformation is not None:
                detections_msg.setTransformation(transformation)

            self._logger.debug(
                f"Created detection message with {len(bboxes)} detections"
            )

            self.out.send(detections_msg)

            self._logger.debug("Detection message sent successfully")

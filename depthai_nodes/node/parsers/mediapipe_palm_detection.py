from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.detection import DetectionParser
from depthai_nodes.node.parsers.utils.medipipe import (
    compute_mediapipe_palm_detections,
    generate_handtracker_anchors,
)


class MPPalmDetectionParser(DetectionParser):
    """Parser class for parsing the output of the Mediapipe Palm detection model. As the
    result, the node sends out the detected hands in the form of a message containing
    bounding boxes, labels, and confidence scores.

    Attributes
    ----------
    output_layer_names: list[str]
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
    **Type**: dai.ImgDetections

    **Description**: dai.ImgDetections message containing bounding boxes, labels, and confidence scores of detected hands.

    See also
    --------
    Official MediaPipe Hands solution:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    """

    def __init__(
        self,
        output_layer_names: list[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_det: int = 100,
        scale: int = 192,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_names: Names of the output layers relevant to the parser.
        @type output_layer_names: list[str]
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

    def setOutputLayerNames(self, output_layer_names: list[str]) -> None:
        """Sets the output layer name(s) for the parser.

        @param output_layer_names: The name of the output layer(s) from which the scores
            are extracted.
        @type output_layer_names: list[str]
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
        head_config: dict[str, Any],
    ) -> "MPPalmDetectionParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: dict[str, Any]
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

            bboxes, scores = self.extract(output)
            bboxes, scores, angles, labels, label_names = self.compute(
                bboxes,
                scores,
                anchors=self._anchors,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                max_det=self.max_det,
                scale=self.scale,
                label_names=self.label_names,
            )
            self.emit(output, bboxes, scores, angles, labels, label_names)

    def extract(self, output: dai.NNData) -> tuple[np.ndarray, np.ndarray]:
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

        if bboxes is None or scores is None:
            raise ValueError("No valid output tensors found.")

        return bboxes.reshape(-1, 18), scores.reshape(-1)

    @staticmethod
    def compute(
        bboxes: np.ndarray,
        scores: np.ndarray,
        *,
        anchors: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
        scale: int,
        label_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
        return compute_mediapipe_palm_detections(
            bboxes,
            scores,
            anchors=anchors,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            scale=scale,
            label_names=label_names,
        )

    def emit(
        self,
        output: dai.NNData,
        bboxes: np.ndarray,
        scores: np.ndarray,
        angles: np.ndarray,
        labels: np.ndarray,
        label_names: list[str] | None,
    ) -> None:
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

        self._logger.debug(f"Created detection message with {len(bboxes)} detections")
        self.out.send(detections_msg)
        self._logger.debug("Detection message sent successfully")

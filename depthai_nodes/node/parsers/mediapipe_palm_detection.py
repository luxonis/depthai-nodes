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

    Attributes:
        output_layer_names (``list[str]``): Names of the output layers relevant to the
            parser.
        conf_threshold (``float``): Confidence score threshold for detected hands.
        iou_threshold (``float``): Non-maximum suppression threshold.
        max_det (``int``): Maximum number of detections to keep.
        scale (``int``): Scale of the input image.

    Note:
        Emits ``dai.ImgDetections`` messages. dai.ImgDetections message containing
        bounding boxes, labels, and confidence scores of detected hands.

    See also:

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

        Args:
            output_layer_names: Names of the output layers relevant to the parser.
            conf_threshold: Confidence score threshold for detected hands.
            iou_threshold: Non-maximum suppression threshold.
            max_det: Maximum number of detections to keep.
            scale: Scale of the input image.
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

        Args:
            output_layer_names: The name of the output layer(s) from which the scores
                are extracted.
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

        Args:
            scale: Scale of the input image.
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

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
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
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
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
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Palm box/landmark predictions reshaped to ``(N, 18)`` and flattened scores,
            selected by their final tensor dimensions.

        Raises:
            ValueError: If no tensors are available or box tensors cannot be reshaped to
                18 values per anchor.
        """
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
        """Compute parser results from extracted tensors without sending messages.

        Args:
            bboxes: Per-anchor palm box and landmark predictions.
            scores: Per-anchor palm score logits.
            anchors: Precomputed anchor coordinates used to decode model predictions.
            conf_threshold: Minimum detection confidence used to filter candidates.
            iou_threshold: Intersection-over-union threshold for non-maximum
                suppression.
            max_det: Maximum number of detection candidates to retain or consider during
                suppression.
            scale: Side length of the square model input in pixels.
            label_names: Optional class-name lookup indexed by predicted class ID.

        Returns:
            Normalized center-XY/width/height boxes, confidence scores, angles in
            degrees, zero-valued class IDs, and optional class names.

        Note:
            Uses
            `depthai_nodes.node.parsers.utils.medipipe.compute_mediapipe_palm_detections`;
            see that helper for tensor layout and validation details.
        """
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
        """Create a ``dai.ImgDetections`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            bboxes: Normalized center-XY/width/height boxes returned by ``compute()``.
            scores: Confidence scores corresponding to the computed payload.
            angles: Rotation angles in degrees corresponding to the boxes.
            labels: Integer class IDs corresponding to the boxes.
            label_names: Optional class names corresponding to the detections.
        """
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

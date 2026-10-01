from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.detection import DetectionParser
from depthai_nodes.node.parsers.utils.ppdet import compute_pp_text_detections


class PPTextDetectionParser(DetectionParser):
    """Parser class for parsing the output of the PaddlePaddle OCR text detection model.

    Attributes:
        output_layer_name (``str``): Name of the output layer relevant to the parser.
        conf_threshold (``float``): The threshold for bounding boxes.
        mask_threshold (``float``): The threshold for the mask.
        max_det (``int``): The maximum number of candidate bounding boxes.

    Output messages:

    **Type**: dai.ImgDetections
    **Description**: dai.ImgDetections message containing bounding boxes and the
    respective confidence scores of detected text.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        conf_threshold: float = 0.5,
        mask_threshold: float = 0.25,
        max_det: int = 100,
    ) -> None:
        """Initializes the parser node.

        Args:
            output_layer_name: Name of the output layer relevant to the parser.
            conf_threshold: The threshold for bounding boxes.
            mask_threshold: The threshold for the mask.
            max_det: The maximum number of candidate bounding boxes.
        """
        super().__init__(
            conf_threshold=conf_threshold,
            iou_threshold=0.5,
            max_det=max_det,
        )
        self.mask_threshold = mask_threshold
        self.output_layer_name = output_layer_name
        self._logger.debug(
            f"PPTextDetectionParser initialized with output_layer_name='{output_layer_name}', conf_threshold={conf_threshold}, mask_threshold={mask_threshold}, max_det={max_det}"
        )

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Sets the name of the output layer.

        Args:
            output_layer_name: The name of the output layer.
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def setMaskThreshold(self, mask_threshold: float = 0.25) -> None:
        """Sets the mask threshold for creating the mask from model output
        probabilities.

        Args:
            mask_threshold: The threshold for the mask.
        """
        if not isinstance(mask_threshold, float):
            raise ValueError("Mask threshold must be a float.")
        self.mask_threshold = mask_threshold
        self._logger.debug(f"Mask threshold set to {self.mask_threshold}")

    def build(self, head_config: dict[str, Any]) -> "PPTextDetectionParser":
        """Configures the parser.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """

        super().build(head_config)
        self.mask_threshold = head_config.get("mask_threshold", self.mask_threshold)

        self._logger.debug(
            f"PPTextDetectionParser built with output_layer_name='{self.output_layer_name}', conf_threshold={self.conf_threshold}, mask_threshold={self.mask_threshold}, max_det={self.max_det}"
        )

        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("PPTextDetectionParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            predictions = self.extract(output)
            bboxes, angles, scores = self.compute(
                predictions,
                mask_threshold=self.mask_threshold,
                conf_threshold=self.conf_threshold,
                max_det=self.max_det,
            )
            self.emit(output, bboxes, angles, scores)

    def extract(self, output: dai.NNData) -> np.ndarray:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Dequantized text probability tensor requested in NCHW storage order.

        Raises:
            ValueError: If no output name is configured and the message does not contain
                exactly one layer, or configured class requirements are not met.
        """
        layers = output.getAllLayerNames()
        self._logger.debug(f"Processing input with layers: {layers}")
        if len(layers) == 1 and self.output_layer_name == "":
            self.output_layer_name = layers[0]
        elif len(layers) != 1 and self.output_layer_name == "":
            raise ValueError(
                f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
            )

        return np.array(
            output.getTensor(
                self.output_layer_name,
                dequantize=True,
                storageOrder=dai.TensorInfo.StorageOrder.NCHW,
            )
        )

    @staticmethod
    def compute(
        predictions: np.ndarray,
        *,
        mask_threshold: float,
        conf_threshold: float,
        max_det: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            predictions: Text probability tensor accepted by
                ``parse_paddle_detection_outputs``.
            mask_threshold: Threshold used to binarize the text probability map.
            conf_threshold: Minimum detection confidence used to filter candidates.
            max_det: Maximum number of detection candidates to retain or consider during
                suppression.

        Returns:
            Normalized center-XY/width/height boxes, rotation angles in degrees, and
            confidence scores.

        Note:
            Uses `depthai_nodes.node.parsers.utils.ppdet.compute_pp_text_detections`;
            see that helper for tensor layout and validation details.
        """
        return compute_pp_text_detections(
            predictions,
            mask_threshold=mask_threshold,
            conf_threshold=conf_threshold,
            max_det=max_det,
        )

    def emit(
        self,
        output: dai.NNData,
        bboxes: np.ndarray,
        angles: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        """Create a ``dai.ImgDetections`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            bboxes: Normalized center-XY/width/height boxes returned by ``compute()``.
            angles: Rotation angles in degrees corresponding to the boxes.
            scores: Confidence scores corresponding to the computed payload.
        """
        message = create_detection_message(bboxes=bboxes, scores=scores, angles=angles)
        message.setTimestamp(output.getTimestamp())
        message.setSequenceNum(output.getSequenceNum())
        message.setTimestampDevice(output.getTimestampDevice())
        transformation = output.getTransformation()
        if transformation is not None:
            message.setTransformation(transformation)

        self._logger.debug(
            f"Created text detection message with {len(bboxes)} detections"
        )
        self.out.send(message)
        self._logger.debug("Text detection message sent successfully")

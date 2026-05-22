from typing import Any, Dict, List, Optional

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils import xyxy_to_xywh


class RFDETRParser(BaseParser):
    """Parser class for parsing the output of the RF-DETR object detection model.

    RF-DETR from Roboflow is a detection transformer model that
    outputs bounding boxes and class probabilities. The model can optionally output
    instance segmentation masks.

    Attributes
    ----------
    conf_threshold : float
        Confidence score threshold for detected objects.
    max_det : int
        Maximum number of detections to keep.
    label_names : Optional[List[str]]
        List of label names for detected objects.
    output_layer_names : List[str]
        Names of the output layers (boxes, logits, and optionally masks).

    Output Message/s
    ----------------
    **Type**: ImgDetectionsExtended

    **Description**: ImgDetectionsExtended message containing bounding boxes, labels,
    confidence scores, and optionally instance segmentation masks.

    References
    ----------
    RF-DETR: https://github.com/roboflow/rf-detr
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        max_det: int = 300,
        label_names: Optional[List[str]] = None,
    ) -> None:
        """Initializes the parser node.

        @param conf_threshold: Confidence score threshold for detected objects.
        @type conf_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        @param label_names: List of label names for detected objects.
        @type label_names: Optional[List[str]]
        """
        super().__init__()
        self.conf_threshold = conf_threshold
        self.max_det = max_det
        self.label_names = label_names
        self.output_layer_names = []
        self._logger.debug(
            f"RFDETRParser initialized with conf_threshold={conf_threshold}, max_det={max_det}"
        )

    @property
    def input(self) -> dai.Node.Input:
        return self._input

    @property
    def out(self) -> dai.Node.Output:
        return self._out

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for detected objects.

        @param threshold: Confidence score threshold for detected objects.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence threshold must be a float.")
        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        self.conf_threshold = threshold
        self._logger.debug(f"Confidence threshold updated to {threshold}")

    def setMaxDetections(self, max_det: int) -> None:
        """Sets the maximum number of detections to keep.

        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        """
        if not isinstance(max_det, int):
            raise ValueError("Max detections must be an integer.")
        if max_det < 1:
            raise ValueError("Max detections must be greater than 0.")
        self.max_det = max_det
        self._logger.debug(f"Maximum detections updated to {max_det}")

    def setLabelNames(self, label_names: List[str]) -> None:
        """Sets the label names for detected objects.

        @param label_names: List of label names for detected objects.
        @type label_names: List[str]
        """
        if not isinstance(label_names, list):
            raise ValueError("Label names must be a list.")
        if not all(isinstance(label, str) for label in label_names):
            raise ValueError("Each label name must be a string.")
        self.label_names = label_names
        self._logger.debug(f"Label names updated to: {label_names}")

    def setOutputLayerNames(self, output_layer_names: List[str]) -> None:
        """Sets the output layer names for the parser.

        @param output_layer_names: List of output layer names.
        @type output_layer_names: List[str]
        """
        if not isinstance(output_layer_names, list):
            raise ValueError("Output layer names must be a list.")
        if not all(isinstance(name, str) for name in output_layer_names):
            raise ValueError("Each output layer name must be a string.")
        self.output_layer_names = output_layer_names
        self._logger.debug(f"Output layer names set to {self.output_layer_names}")

    def build(self, head_config: Dict[str, Any]) -> "RFDETRParser":
        """Configures the parser based on the head configuration.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: RFDETRParser
        """
        self.conf_threshold = head_config.get("conf_threshold", self.conf_threshold)
        self.max_det = head_config.get("max_det", self.max_det)
        self.label_names = head_config.get("classes", self.label_names)
        self.output_layer_names = head_config.get("outputs", self.output_layer_names)

        self._logger.debug(
            f"RFDETRParser built with conf_threshold={self.conf_threshold}, max_det={self.max_det}"
        )
        return self

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation function.

        @param x: Input array.
        @type x: np.ndarray
        @return: Sigmoid of input.
        @rtype: np.ndarray
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from (cx, cy, w, h) format to (xmin, ymin, xmax, ymax) format.

        Boxes are expected to be in normalized coordinates [0, 1].

        @param boxes: Boxes in cxcywh format, shape (..., 4)
        @type boxes: np.ndarray
        @return: Boxes in xyxy format, shape (..., 4)
        @rtype: np.ndarray
        """
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        return np.stack([xmin, ymin, xmax, ymax], axis=-1)

    def run(self):
        self._logger.debug("RFDETRParser run started")

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            # Get output layers
            layer_names = self.output_layer_names or output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {layer_names}")

            if len(layer_names) < 2 or len(layer_names) > 3:
                raise ValueError(
                    f"Expected 2 or 3 output layers (boxes, logits, optional masks), got {len(layer_names)} layers."
                )

            # outputs[0]: boxes in cxcywh format (normalized coordinates [0, 1])
            # outputs[1]: class logits (need sigmoid activation)
            # outputs[2]: instance segmentation masks (optional)
            boxes_tensor = output.getTensor(layer_names[0], dequantize=True).astype(
                np.float32
            )
            logits_tensor = output.getTensor(layer_names[1], dequantize=True).astype(
                np.float32
            )

            masks_tensor = None
            if len(layer_names) == 3:
                masks_tensor = output.getTensor(layer_names[2], dequantize=True).astype(
                    np.float32
                )

            prob = self._sigmoid(logits_tensor)

            scores = np.max(prob, axis=2).squeeze()  # (num_queries,)
            labels = np.argmax(prob, axis=2).squeeze()  # (num_queries,)

            sorted_idx = np.argsort(scores)[::-1]
            scores = scores[sorted_idx][: self.max_det]
            labels = labels[sorted_idx][: self.max_det]
            boxes = boxes_tensor.squeeze()[sorted_idx][: self.max_det]

            if masks_tensor is not None:
                masks = masks_tensor.squeeze()[sorted_idx][: self.max_det]

            # Convert boxes from cxcywh (normalized) to xyxy (normalized)
            boxes = self._box_cxcywh_to_xyxy(boxes)

            # Filter detections by confidence threshold
            confidence_mask = scores > self.conf_threshold
            scores = scores[confidence_mask]
            labels = labels[confidence_mask]
            boxes = boxes[confidence_mask]

            if masks_tensor is not None:
                masks = masks[confidence_mask]

            boxes = xyxy_to_xywh(boxes)
            boxes = np.clip(boxes, 0, 1)

            label_names_list = None
            if self.label_names:
                label_names_list = [
                    (
                        self.label_names[int(label)]
                        if int(label) < len(self.label_names)
                        else f"class_{int(label)}"
                    )
                    for label in labels
                ]

            # Create detection message
            message = create_detection_message(
                bboxes=boxes,
                scores=scores,
                labels=labels.astype(int),
                label_names=label_names_list,
            )

            # Set message metadata
            transformation = output.getTransformation()
            if transformation is not None:
                message.setTransformation(transformation)
            message.setTimestamp(output.getTimestamp())
            message.setSequenceNum(output.getSequenceNum())
            message.setTimestampDevice(output.getTimestampDevice())

            self._logger.debug(f"Created detections message with {len(boxes)} objects")
            self.out.send(message)
            self._logger.debug("Detections message sent successfully")

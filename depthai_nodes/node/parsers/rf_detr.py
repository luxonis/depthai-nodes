from typing import Any

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils.rf_detr import (
    compute_rfdetr_detections,
)


class RFDETRParser(BaseParser):
    """Parser class for parsing the output of the RF-DETR object detection model.

    RF-DETR from Roboflow is a detection transformer model that outputs bounding boxes
    and class probabilities. The model can optionally output instance segmentation
    masks.

    Attributes:
        conf_threshold (``float``): Confidence score threshold for detected objects.
        max_det (``int``): Maximum number of detections to keep.
        label_names (``list[str] | None``): List of label names for detected objects.
        mask_conf (``float``): Confidence threshold for binarizing instance segmentation
            masks.
        output_layer_names (``list[str]``): Names of the output layers (boxes, logits,
            and optionally masks).

    Note:
        Emits ``dai.ImgDetections`` messages. dai.ImgDetections message containing
        bounding boxes, labels, confidence scores, and optionally instance segmentation
        masks.

    References:

    RF-DETR: https://github.com/roboflow/rf-detr
    """

    _DET_MODE = 0
    _SEG_MODE = 1

    def __init__(
        self,
        conf_threshold: float = 0.5,
        max_det: int = 300,
        label_names: list[str] | None = None,
        mask_conf: float = 0.5,
    ) -> None:
        """Initializes the parser node.

        Args:
            conf_threshold: Confidence score threshold for detected objects.
            max_det: Maximum number of detections to keep.
            label_names: List of label names for detected objects.
            mask_conf: Mask confidence threshold for instance segmentation masks.
        """
        super().__init__()
        self.conf_threshold = conf_threshold
        self.max_det = max_det
        self.label_names = label_names
        self.mask_conf = mask_conf
        self.output_layer_names: list[str] = []
        self.input_shape: tuple[int, int] | None = None
        self._logger.debug(
            f"RFDETRParser initialized with conf_threshold={conf_threshold}, max_det={max_det}, mask_conf={mask_conf}"
        )

    @property
    def input(self) -> dai.Node.Input:
        """Input port accepting ``dai.NNData``."""
        return self._input

    @property
    def out(self) -> dai.Node.Output:
        """Output port carrying parsed messages."""
        return self._out

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for detected objects.

        Args:
            threshold: Confidence score threshold for detected objects.
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence threshold must be a float.")
        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        self.conf_threshold = threshold
        self._logger.debug(f"Confidence threshold updated to {threshold}")

    def setMaxDetections(self, max_det: int) -> None:
        """Sets the maximum number of detections to keep.

        Args:
            max_det: Maximum number of detections to keep.
        """
        if not isinstance(max_det, int):
            raise ValueError("Max detections must be an integer.")
        if max_det < 1:
            raise ValueError("Max detections must be greater than 0.")
        self.max_det = max_det
        self._logger.debug(f"Maximum detections updated to {max_det}")

    def setLabelNames(self, label_names: list[str]) -> None:
        """Sets the label names for detected objects.

        Args:
            label_names: List of label names for detected objects.
        """
        if not isinstance(label_names, list):
            raise ValueError("Label names must be a list.")
        if not all(isinstance(label, str) for label in label_names):
            raise ValueError("Each label name must be a string.")
        self.label_names = label_names
        self._logger.debug(f"Label names updated to: {label_names}")

    def setMaskConfidence(self, mask_conf: float) -> None:
        """Sets the mask confidence threshold.

        Args:
            mask_conf: The mask confidence threshold.
        """
        if not isinstance(mask_conf, float):
            raise ValueError("Mask confidence threshold must be a float.")

        if mask_conf < 0 or mask_conf > 1:
            raise ValueError("Mask confidence threshold must be between 0 and 1.")

        self.mask_conf = mask_conf
        self._logger.debug(f"Mask confidence threshold updated to {mask_conf}")

    def setOutputLayerNames(self, output_layer_names: list[str]) -> None:
        """Sets the output layer names for the parser.

        Args:
            output_layer_names: List of output layer names.
        """
        if not isinstance(output_layer_names, list):
            raise ValueError("Output layer names must be a list.")
        if not all(isinstance(name, str) for name in output_layer_names):
            raise ValueError("Each output layer name must be a string.")
        self.output_layer_names = output_layer_names
        self._logger.debug(f"Output layer names set to {self.output_layer_names}")

    def build(self, head_config: dict[str, Any]) -> "RFDETRParser":
        """Configures the parser based on the head configuration.

        Args:
            head_config: The head configuration for the parser.

        Returns:
            The parser object with the head configuration set.
        """
        self.conf_threshold = head_config.get("conf_threshold", self.conf_threshold)
        self.max_det = head_config.get("max_det", self.max_det)
        self.label_names = head_config.get("classes", self.label_names)
        self.mask_conf = head_config.get("mask_conf", self.mask_conf)
        self.output_layer_names = head_config.get("outputs", self.output_layer_names)

        inputs = head_config.get("model_inputs", [])
        if inputs:
            input_shape = inputs[0].get("shape")
            input_layout = inputs[0].get("layout")

            if input_shape and input_layout:
                if input_layout == "NCHW":
                    self.input_shape = (input_shape[2], input_shape[3])
                elif input_layout == "NHWC":
                    self.input_shape = (input_shape[1], input_shape[2])
                else:
                    raise ValueError(f"Unsupported input layout: {input_layout}")

        if self.output_layer_names and len(self.output_layer_names) not in (2, 3):
            raise ValueError(
                f"RFDETRParser expects 2 outputs for detection or 3 outputs for "
                f"segmentation, got {len(self.output_layer_names)} outputs: "
                f"{self.output_layer_names}."
            )

        self._logger.debug(
            f"RFDETRParser built with conf_threshold={self.conf_threshold}, "
            f"max_det={self.max_det}, mask_conf={self.mask_conf}, "
            f"input_shape={self.input_shape}, outputs={self.output_layer_names}"
        )
        return self

    def run(self):
        """Read queued network outputs, parse them, and emit results while running.

        The pipeline invokes this processing loop. It exits when the input queue closes
        or the node stops.
        """
        self._logger.debug("RFDETRParser run started")

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            boxes_tensor, logits_tensor, masks_tensor = self.extract(output)
            boxes, scores, labels, label_names_list, final_mask = self.compute(
                boxes_tensor,
                logits_tensor,
                conf_threshold=self.conf_threshold,
                max_det=self.max_det,
                label_names=self.label_names,
                mask_conf=self.mask_conf,
                input_shape=self.input_shape,
                masks_tensor=masks_tensor,
            )
            self.emit(output, boxes, scores, labels, label_names_list, final_mask)

    def extract(
        self, output: dai.NNData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Select and dequantize the model tensors needed for parsing.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.

        Returns:
            Float32 box, class-logit, and optional mask tensors in configured layer
            order. Without a mask layer, the third value is ``None``.

        Raises:
            ValueError: If the selected outputs do not contain two or three layers.
        """
        layer_names = self.output_layer_names or output.getAllLayerNames()
        self._logger.debug(f"Processing input with layers: {layer_names}")

        if len(layer_names) < 2 or len(layer_names) > 3:
            raise ValueError(
                "Expected 2 or 3 output layers "
                f"(boxes, logits, optional masks), got {len(layer_names)} layers."
            )

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
        return boxes_tensor, logits_tensor, masks_tensor

    def compute(
        self,
        boxes_tensor: np.ndarray,
        logits_tensor: np.ndarray,
        *,
        conf_threshold: float,
        max_det: int,
        label_names: list[str] | None,
        mask_conf: float,
        input_shape: tuple[int, int] | None,
        masks_tensor: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None, np.ndarray | None]:
        """Compute parser results from extracted tensors without sending messages.

        Args:
            boxes_tensor: Batched normalized center-XY/width/height predictions.
            logits_tensor: Class logits of shape ``(1, queries, classes)``.
            conf_threshold: Minimum detection confidence used to filter candidates.
            max_det: Maximum number of detection candidates to retain or consider during
                suppression.
            label_names: Optional class-name lookup indexed by predicted class ID.
            mask_conf: Probability threshold used to binarize mask logits.
            input_shape: Model input image shape as ``(height, width)``.
            masks_tensor: Optional per-query mask logits, ordered like the box
                predictions.

        Returns:
            Boxes in normalized center-XY/width/height format, scores, integer class
            IDs, optional class names, and an optional HW uint8 instance mask. Mask
            values index returned detections; 255 is background. Segmentation retains at
            most 255 instances, and higher-confidence masks win overlaps.

        Note:
            Uses `depthai_nodes.node.parsers.utils.rf_detr.compute_rfdetr_detections`;
            see that helper for tensor layout and validation details.
        """
        return compute_rfdetr_detections(
            boxes_tensor,
            logits_tensor,
            conf_threshold=conf_threshold,
            max_det=max_det,
            label_names=label_names,
            mask_conf=mask_conf,
            input_shape=input_shape,
            masks_tensor=masks_tensor,
            logger=self._logger,
        )

    def emit(
        self,
        output: dai.NNData,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        label_names_list: list[str] | None,
        final_mask: np.ndarray | None,
    ) -> None:
        """Create a ``dai.ImgDetections`` message and send it on ``out``.

        Copies source timestamps and sequence number, and carries the source image
        transformation when present.

        Args:
            output: Neural network output carrying tensors and source timestamps,
                sequence number, and optional image transformation.
            boxes: Normalized center-XY/width/height boxes returned by ``compute()``.
            scores: Confidence scores corresponding to the computed payload.
            labels: Integer class IDs corresponding to the boxes.
            label_names_list: Optional class names corresponding to the detections.
            final_mask: Optional instance mask whose IDs index the returned detections;
                255 is background.
        """
        message = create_detection_message(
            bboxes=boxes,
            scores=scores,
            labels=labels.astype(int),
            label_names=label_names_list,
            masks=final_mask,
        )

        transformation = output.getTransformation()
        if transformation is not None:
            message.setTransformation(transformation)
        message.setTimestamp(output.getTimestamp())
        message.setSequenceNum(output.getSequenceNum())
        message.setTimestampDevice(output.getTimestampDevice())

        self._logger.debug(f"Created detections message with {len(boxes)} objects")
        self.out.send(message)
        self._logger.debug("Detections message sent successfully")

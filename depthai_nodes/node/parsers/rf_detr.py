from typing import Any

import cv2
import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.base_parser import BaseParser
from depthai_nodes.node.parsers.utils import xyxy_to_xywh
from depthai_nodes.node.parsers.utils.masks_utils import crop_mask


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
    label_names : list[str] | None
        List of label names for detected objects.
    mask_conf : float
        Confidence threshold for binarizing instance segmentation masks.
    output_layer_names : list[str]
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

        @param conf_threshold: Confidence score threshold for detected objects.
        @type conf_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        @param label_names: List of label names for detected objects.
        @type label_names: list[str] | None
        @param mask_conf: Mask confidence threshold for instance segmentation masks.
        @type mask_conf: float
        """
        super().__init__()
        self.conf_threshold = conf_threshold
        self.max_det = max_det
        self.label_names = label_names
        self.mask_conf = mask_conf
        self.output_layer_names: list[str] = []
        self.input_shape: tuple[int, int] | None = None
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

    def setLabelNames(self, label_names: list[str]) -> None:
        """Sets the label names for detected objects.

        @param label_names: List of label names for detected objects.
        @type label_names: list[str]
        """
        if not isinstance(label_names, list):
            raise ValueError("Label names must be a list.")
        if not all(isinstance(label, str) for label in label_names):
            raise ValueError("Each label name must be a string.")
        self.label_names = label_names
        self._logger.debug(f"Label names updated to: {label_names}")

    def setMaskConfidence(self, mask_conf: float) -> None:
        """Set mask confidence threshold.

        RF-DETR Seg outputs mask logits. The parser applies sigmoid to the mask logits,
        so this value is interpreted as a normal probability threshold.
        """
        if not isinstance(mask_conf, float):
            raise ValueError("Mask confidence threshold must be a float.")

        if mask_conf < 0 or mask_conf > 1:
            raise ValueError("Mask confidence threshold must be between 0 and 1.")

        self.mask_conf = mask_conf
        self._logger.debug(f"Mask confidence threshold updated to {mask_conf}")

    def setOutputLayerNames(self, output_layer_names: list[str]) -> None:
        """Sets the output layer names for the parser.

        @param output_layer_names: List of output layer names.
        @type output_layer_names: list[str]
        """
        if not isinstance(output_layer_names, list):
            raise ValueError("Output layer names must be a list.")
        if not all(isinstance(name, str) for name in output_layer_names):
            raise ValueError("Each output layer name must be a string.")
        self.output_layer_names = output_layer_names
        self._logger.debug(f"Output layer names set to {self.output_layer_names}")

    def build(self, head_config: dict[str, Any]) -> "RFDETRParser":
        """Configures the parser based on the head configuration.

        @param head_config: The head configuration for the parser.
        @type head_config: dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: RFDETRParser
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

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation function.

        @param x: Input array.
        @type x: np.ndarray
        @return: Sigmoid of input.
        @rtype: np.ndarray
        """
        x = np.asarray(x, dtype=np.float32)
        result = np.empty_like(x, dtype=np.float32)

        positive_mask = x >= 0
        result[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))

        exp_x = np.exp(x[~positive_mask])
        result[~positive_mask] = exp_x / (1.0 + exp_x)

        return result

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

    def _process_mask(
        self,
        mask_logits: np.ndarray,
        bbox: np.ndarray,
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        """Process a single RF-DETR Seg mask.

        @param mask_logits: Mask logits in [H, W] format.
        @type mask_logits: np.ndarray
        @param bbox: Bounding box in normalized cxcywh format.
        @type bbox: np.ndarray
        @param input_shape: Target input shape in (height, width) format.
        @type input_shape: tuple[int, int]
        @return: Binary mask resized to input shape.
        @rtype: np.ndarray
        """

        if mask_logits.ndim != 2:
            raise ValueError(
                f"Expected mask logits of shape (H, W), got {mask_logits.shape}."
            )

        mask_h, mask_w = mask_logits.shape

        scaled_bbox = bbox * np.array([mask_w, mask_h, mask_w, mask_h])
        mask = self._sigmoid(mask_logits)
        mask = crop_mask(mask, scaled_bbox)
        mask = (mask > self.mask_conf).astype(np.uint8)

        return cv2.resize(
            mask,
            (input_shape[1], input_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

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
                    "Expected 2 or 3 output layers "
                    f"(boxes, logits, optional masks), got {len(layer_names)} layers."
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
            boxes_cxcywh = boxes_tensor.squeeze()[sorted_idx][: self.max_det]

            masks = None
            if masks_tensor is not None:
                masks = masks_tensor.squeeze()[sorted_idx][: self.max_det]

            # Convert boxes from cxcywh (normalized) to xyxy (normalized)
            boxes = np.clip(self._box_cxcywh_to_xyxy(boxes_cxcywh), 0, 1)

            # Filter detections by confidence threshold
            confidence_mask = scores > self.conf_threshold
            scores = scores[confidence_mask]
            labels = labels[confidence_mask]
            boxes = boxes[confidence_mask]
            boxes_cxcywh = boxes_cxcywh[confidence_mask]

            if masks is not None:
                masks = masks[confidence_mask]

            final_mask = None
            mode = self._SEG_MODE if masks is not None else self._DET_MODE

            if mode == self._SEG_MODE:
                if self.input_shape is None:
                    raise ValueError(
                        "RFDETRParser segmentation mode requires model input shape."
                    )

                final_mask = np.full(self.input_shape, 255, dtype=np.uint8)

                for i, (mask_logits, bbox) in enumerate(zip(masks, boxes_cxcywh)):
                    resized_mask = self._process_mask(
                        mask_logits,
                        bbox,
                        self.input_shape,
                    )
                    final_mask[resized_mask > 0] = i

            boxes = xyxy_to_xywh(boxes)

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
                masks=final_mask,
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

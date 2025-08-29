from typing import Any, Dict

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.detection import DetectionParser
from depthai_nodes.node.parsers.utils.ppdet import parse_paddle_detection_outputs


class PPTextDetectionParser(DetectionParser):
    """Parser class for parsing the output of the PaddlePaddle OCR text detection model.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    conf_threshold : float
        The threshold for bounding boxes.
    mask_threshold : float
        The threshold for the mask.
    max_det : int
        The maximum number of candidate bounding boxes.

    Output Message/s
    -------
    **Type**: dai.ImgDetections
    **Description**: ImgDetections message containing bounding boxes and the respective confidence scores of detected text.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        conf_threshold: float = 0.5,
        mask_threshold: float = 0.25,
        max_det: int = 100,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param conf_threshold: The threshold for bounding boxes.
        @type conf_threshold: float
        @param mask_threshold: The threshold for the mask.
        @type mask_threshold: float
        @param max_det: The maximum number of candidate bounding boxes.
        @type max_det:
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

        @param output_layer_name: The name of the output layer.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name
        self._logger.debug(f"Output layer name set to '{self.output_layer_name}'")

    def setMaskThreshold(self, mask_threshold: float = 0.25) -> None:
        """Sets the mask threshold for creating the mask from model output
        probabilities.

        @param threshold: The threshold for the mask.
        @type threshold: float
        """
        if not isinstance(mask_threshold, float):
            raise ValueError("Mask threshold must be a float.")
        self.mask_threshold = mask_threshold
        self._logger.debug(f"Mask threshold set to {self.mask_threshold}")

    def build(self, head_config: Dict[str, Any]) -> "PPTextDetectionParser":
        """Configures the parser.

        @param config: The head configuration for the parser.
        @type config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: PPTextDetectionParser
        """

        super().build(head_config)
        self.mask_threshold = head_config.get("mask_threshold", self.mask_threshold)

        self._logger.debug(
            f"PPTextDetectionParser built with output_layer_name='{self.output_layer_name}', conf_threshold={self.conf_threshold}, mask_threshold={self.mask_threshold}, max_det={self.max_det}"
        )

        return self

    def run(self):
        self._logger.debug("PPTextDetectionParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            layers = output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {layers}")
            if len(layers) == 1 and self.output_layer_name == "":
                self.output_layer_name = layers[0]
            elif len(layers) != 1 and self.output_layer_name == "":
                raise ValueError(
                    f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
                )

            predictions = np.array(
                output.getTensor(
                    self.output_layer_name,
                    dequantize=True,
                    storageOrder=dai.TensorInfo.StorageOrder.NCHW,
                )
            )

            _, _, height, width = predictions.shape

            bboxes, angles, scores = parse_paddle_detection_outputs(
                predictions,
                self.mask_threshold,
                self.conf_threshold,
                self.max_det,
                width=width,
                height=height,
            )
            message = create_detection_message(
                bboxes=bboxes, scores=scores, angles=angles
            )
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

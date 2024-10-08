from typing import Any, Dict

import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .detection import DetectionParser
from .utils import parse_paddle_detection_outputs


class PPTextDetectionParser(DetectionParser):
    """Parser class for parsing the output of the PaddlePaddle OCR text detection model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    mask_threshold : float
        The threshold for the mask.
    bbox_threshold : float
        The threshold for bounding boxes.
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
        """Initializes the PPTextDetectionParser node.

        @param output_layer_name: The name of the output layer from which the scores are
            extracted.
        @type output_layer_name: str
        @param mask_threshold: The threshold for the mask.
        @type mask_threshold: float
        @param conf_threshold: The threshold for bounding boxes.
        @type conf_threshold: float
        @param max_det: The maximum number of candidate bounding boxes.
        @type max_det:
        """
        super().__init__(
            output_layer_name=output_layer_name,
            conf_threshold=conf_threshold,
            max_det=max_det,
        )
        self.mask_threshold = mask_threshold

    def setMaskThreshold(self, mask_threshold: float = 0.25) -> None:
        """Sets the mask threshold for creating the mask from model output
        probabilities.

        @param threshold: The threshold for the mask.
        @type threshold: float
        """
        self.mask_threshold = mask_threshold

    def build(self, head_config: Dict[str, Any]) -> "PPTextDetectionParser":
        """Sets the head configuration for the parser. If any configuration parameters
        are missing, default values are used.

        Attributes
        ----------
        config : Dict
            The head configuration for the parser.

        Returns
        -------
        PPTextDetectionParser
            Returns the parser object with the head configuration set.
        """
        super().build(head_config)
        self.mask_threshold = head_config.get("mask_threshold", 0.25)

        return self

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            layers = output.getAllLayerNames()

            if len(layers) == 1 and self.output_layer_name == "":
                self.output_layer_name = layers[0]
            elif len(layers) != 1 and self.output_layer_name == "":
                raise ValueError(
                    f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
                )

            predictions = np.array(
                output.getTensor(self.output_layer_name, dequantize=True)
            )

            bboxes, angles, corners, scores = parse_paddle_detection_outputs(
                predictions,
                self.mask_threshold,
                self.conf_threshold,
                self.max_det,
            )

            message = create_detection_message(
                bboxes, scores, angles=angles, keypoints=corners
            )
            message.setTimestamp(output.getTimestamp())

            self.out.send(message)

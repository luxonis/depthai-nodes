import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .base_parser import BaseParser


class DetectionParser(BaseParser):
    """Parser class for parsing the output of a detection model. The parser expects the
    output of the model to be in the (x_min, y_min, x_max, y_max, confidence) format. As
    the result, the node sends out the detected objects in the form of a message
    containing bounding boxes and confidence scores.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    conf_threshold : float
        Confidence score threshold of detected bounding boxes.
    iou_threshold : float
        Non-maximum suppression threshold.
    max_det : int
        Maximum number of detections to keep.

    Output Message/s
        -------
        **Type**: ImgDetectionsExtended

        **Description**: ImgDetectionsExtended message containing bounding boxes and confidence scores of detected objects.
    ----------------
    """

    def __init__(
        self,
        output_layer_name: str = "",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_det: int = 100,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param conf_threshold: Confidence score threshold of detected bounding boxes.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Sets the output layer name(s) for the parser.

        @param output_layer_name: The name of the output layer(s) from which the scores
            are extracted.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for detected objects.

        @param threshold: Confidence score threshold for detected objects.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence threshold must be a float.")
        self.conf_threshold = threshold

    def setIOUThreshold(self, threshold: float) -> None:
        """Sets the non-maximum suppression threshold.

        @param threshold: Non-maximum suppression threshold.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("IOU threshold must be a float.")
        self.iou_threshold = threshold

    def setMaxDetections(self, max_det: int) -> None:
        """Sets the maximum number of detections to keep.

        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        """
        if not isinstance(max_det, int):
            raise ValueError("Max detections must be an integer.")
        self.max_det = max_det

    def build(self, head_config) -> "DetectionParser":
        """Configures the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        DetectionParser
            Returns the parser object with the head configuration set.
        """

        self.conf_threshold = head_config.get("conf_threshold", self.conf_threshold)
        self.iou_threshold = head_config.get("iou_threshold", self.iou_threshold)
        self.max_det = head_config.get("max_det", self.max_det)

        return self

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break

            layers = output.getAllLayerNames()

            if len(layers) == 1 and self.output_layer_names == "":
                self.output_layer_names = layers[0]
            elif len(layers) != 1 and self.output_layer_names == "":
                raise ValueError(
                    f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
                )

            predictions = np.array(
                output.getTensor(self.output_layer_names, dequantize=True)
            )

            if len(predictions) == 0:
                message = create_detection_message(predictions, np.array([]))
            else:
                message = create_detection_message(
                    predictions[:, :4], predictions[:, 4]
                )

            message.setTimestamp(output.getTimestamp())

            self.out.send(message)

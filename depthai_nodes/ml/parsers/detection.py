import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .base_parser import BaseParser


class DetectionParser(BaseParser):
    def __init__(
        self,
        output_layer_name: str = "",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
    ) -> None:
        """Parser class for parsing the output of a detection model. The parser expects
        the output of the model to be in the (x_min, y_min, x_max, y_max, confidence)
        format. As the result, the node sends out the detected objects in the form of a
        message containing bounding boxes and confidence scores.

        Attributes
        ----------
        input : Node.Input
            Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
        out : Node.Output
            Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.Parser sends the processed network results to this output in form of messages. It is a linking point from which the processed network results are retrieved.
        @param output_layer_name: The name of the output layer(s) from which the scores are extracted.
        @type output_layer_name: Union[str, List[str]]
        @param conf_threshold: Confidence score threshold of detected bounding boxes.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_detections: Maximum number of detections to keep.
        @type max_detections: int

        Output Message/s
        -------
        **Type**: ImgDetectionsExtended

        **Description**: ImgDetectionsExtended message containing bounding boxes and confidence scores of detected objects.
        """
        super().__init__()
        self.output_layer_name = output_layer_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Sets the output layer name(s) for the parser.

        @param output_layer_name: The name of the output layer(s) from which the scores
            are extracted.
        @type output_layer_name: Union[str, List[str]]
        """
        self.output_layer_name = output_layer_name

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for detected objects.

        @param threshold: Confidence score threshold for detected objects.
        @type threshold: float
        """
        self.conf_threshold = threshold

    def setIOUThreshold(self, threshold: float) -> None:
        """Sets the non-maximum suppression threshold.

        @param threshold: Non-maximum suppression threshold.
        @type threshold: float
        """
        self.iou_threshold = threshold

    def setMaxDetections(self, max_det: int) -> None:
        """Sets the maximum number of detections to keep.

        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        """
        self.max_det = max_det

    def build(self, head_config) -> "DetectionParser":
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        DetectionParser
            Returns the parser object with the head configuration set.
        """

        output_layer = head_config["outputs"]
        if len(output_layer) != 1:
            raise ValueError(
                f"Only one output layer supported for Detection, got {output_layer} layers."
            )

        self.output_layer_names = output_layer[0]
        self.conf_threshold = head_config.get("conf_threshold", 0.5)
        self.iou_threshold = head_config.get("iou_threshold", 0.5)
        self.max_detections = head_config.get("max_detections", 100)

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


def xywh2xyxy(bboxes: np.ndarray):
    """Convert bounding box coordinates from (x, y, width, height) to (x_min, y_min,
    x_max, y_max).

    @param bboxes: A numpy array of shape (N, 4) containing the bounding boxes in (x, y, width, height) format.
    @type np.ndarray
    @return: A numpy array of shape (N, 4) containing the bounding boxes in (x_min, y_min, x_max, y_max) format.
    @type np.ndarray
    """

    xyxy_bboxes = np.zeros_like(bboxes)
    xyxy_bboxes[:, 0] = bboxes[:, 0]  # x_min = x
    xyxy_bboxes[:, 1] = bboxes[:, 1]  # y_min = y
    xyxy_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max = x + w
    xyxy_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max = y + h
    return xyxy_bboxes

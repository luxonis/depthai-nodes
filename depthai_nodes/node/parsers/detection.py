import depthai as dai
import numpy as np

from depthai_nodes.message.creators import (
    create_detection_message,
)
from depthai_nodes.node.parsers.utils.detection import compute_detection_outputs

from .base_parser import BaseParser


class DetectionParser(BaseParser):
    """Parser class for parsing the output of a "general" detection model. The parser expects the
    output of the model to have two tensors: one for bounding boxes and one for scores.
    Tensor for bboxes should be of shape (N, 4) and scores should be of shape (N,).
    Bboxes are expected to be in the format [xmin, ymin, xmax, ymax]. If this is not the case you can check other parsers
    or create a new one. As the result, the node sends out the detected objects in the form of a message
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
    label_names : list[str]
    List of label names for detected objects.

    Output Message/s
        -------
        **Type**: dai.ImgDetections

        **Description**: dai.ImgDetections message containing bounding boxes and confidence scores of detected objects.
    ----------------
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_det: int = 100,
        label_names: list[str] | None = None,
    ) -> None:
        """Initializes the parser node.

        @param conf_threshold: Confidence score threshold of detected bounding boxes.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        """
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.label_names = label_names
        self._logger.debug(
            f"DetectionParser initialized with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}, max_det={max_det}"
        )

    def setConfidenceThreshold(self, threshold: float) -> None:
        """Sets the confidence score threshold for detected objects.

        @param threshold: Confidence score threshold for detected objects.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("Confidence threshold must be a float.")
        self.conf_threshold = threshold
        self._logger.debug(f"Confidence threshold updated to {threshold}")

    def setIouThreshold(self, threshold: float) -> None:
        """Sets the non-maximum suppression threshold.

        @param threshold: Non-maximum suppression threshold.
        @type threshold: float
        """
        if not isinstance(threshold, float):
            raise ValueError("IOU threshold must be a float.")
        self.iou_threshold = threshold
        self._logger.debug(f"IoU threshold updated to {threshold}")

    def setMaxDetections(self, max_det: int) -> None:
        """Sets the maximum number of detections to keep.

        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        """
        if not isinstance(max_det, int):
            raise ValueError("Max detections must be an integer.")
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

    def build(self, head_config) -> "DetectionParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: DetectionParser
        """

        self.conf_threshold = head_config.get("conf_threshold", self.conf_threshold)
        self.iou_threshold = head_config.get("iou_threshold", self.iou_threshold)
        self.max_det = head_config.get("max_det", self.max_det)
        self.label_names = head_config.get("classes", self.label_names)

        self._logger.debug(
            f"DetectionParser built with conf_threshold={self.conf_threshold}, iou_threshold={self.iou_threshold}, max_det={self.max_det}"
        )
        return self

    def run(self):
        self._logger.debug("DetectionParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            bboxes, scores = self.extract(output)
            bboxes, scores = self.compute(
                bboxes,
                scores,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                max_det=self.max_det,
            )
            self.emit(output, bboxes, scores)

    def extract(self, output: dai.NNData) -> tuple[np.ndarray, np.ndarray]:
        layers = output.getAllLayerNames()
        self._logger.debug(f"Processing input with layers: {layers}")
        if len(layers) != 2:
            raise ValueError(
                f"Expected 2 output layers, got {len(layers)} layers. Please use different parser or create a new one."
            )

        bboxes = None
        scores = None

        for layer in layers:
            tensor = np.asarray(output.getTensor(layer, dequantize=True))
            if tensor.shape[-1] == 4 and tensor.ndim != 1:
                bboxes = tensor
            else:
                scores = tensor

        if bboxes is None or scores is None:
            raise ValueError(
                "Bounding boxes or scores are missing in the output. Please check the NN model."
            )

        return bboxes.reshape(-1, 4), scores.reshape(-1)

    @staticmethod
    def compute(
        bboxes: np.ndarray,
        scores: np.ndarray,
        *,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_detection_outputs(
            bboxes,
            scores,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )

    def emit(self, output: dai.NNData, bboxes: np.ndarray, scores: np.ndarray) -> None:
        message = create_detection_message(
            bboxes=bboxes, scores=scores, label_names=self.label_names
        )
        transformation = output.getTransformation()
        if transformation is not None:
            message.setTransformation(transformation)
        message.setTimestamp(output.getTimestamp())
        message.setSequenceNum(output.getSequenceNum())
        message.setTimestampDevice(output.getTimestampDevice())

        self._logger.debug(f"Created detections message with {len(bboxes)} objects")
        self.out.send(message)
        self._logger.debug("Detections message sent successfully")

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
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
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
        """
        Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]

        @return: The parser object with the head configuration set.
        @rtype: DetectionParser
        """

        self.conf_threshold = head_config.get("conf_threshold", self.conf_threshold)
        self.iou_threshold = head_config.get("iou_threshold", self.iou_threshold)
        self.max_det = head_config.get("max_det", self.max_det)

        return self

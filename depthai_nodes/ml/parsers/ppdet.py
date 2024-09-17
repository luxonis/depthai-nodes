import depthai as dai

from ..messages.creators import create_corner_detection_message
from .utils import parse_paddle_detection_outputs


class PPTextDetectionParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the PP-OCR text detection model.

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
    max_detections : int
        The maximum number of candidate bounding boxes.

    Output Message/s
    -------
    **Type**: dai.ImgDetections
    **Description**: ImgDetections message containing bounding boxes and the respective confidence scores of detected text.
    """

    def __init__(
        self,
        mask_threshold: float = 0.3,
        bbox_threshold: float = 0.7,
        max_detections: int = 1000,
    ):
        """Initializes the PPTextDetectionParser node.

        @param mask_threshold: The threshold for the mask.
        @type mask_threshold: float
        @param bbox_threshold: The threshold for bounding boxes.
        @type bbox_threshold: float
        @param max_detections: The maximum number of candidate bounding boxes.
        @type max_detections:
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.mask_threshold = mask_threshold
        self.bbox_threshold = bbox_threshold
        self.max_detections = max_detections

    def setMaskThreshold(self, mask_threshold: float = 0.3):
        """Sets the mask threshold for creating the mask from model output
        probabilities.

        @param threshold: The threshold for the mask.
        @type threshold: float
        """
        self.mask_threshold = mask_threshold

    def setBoundingBoxThreshold(self, bbox_threshold: float = 0.7):
        """Sets the threshold for bounding boxes confidences.

        @param threshold: The threshold for bounding box confidences.
        @type threshold: float
        """
        self.bbox_threshold = bbox_threshold

    def setMaxDetections(self, max_detections: int = 1000):
        """Sets the maximum number of candidate bounding boxes. Recommended upper limit
        is 1000.

        @param max_detections: The maximum number of candidate bounding boxes.
        @type max_detections: int
        """
        self.max_detections = max_detections

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            predictions = output.getFirstTensor()

            bboxes, scores = parse_paddle_detection_outputs(
                predictions,
                self.mask_threshold,
                self.bbox_threshold,
                self.max_detections,
            )

            # bboxes = corners2xyxy(bboxes)

            message = create_corner_detection_message(bboxes, scores)
            message.setTimestamp(output.getTimestamp())

            self.out.send(message)

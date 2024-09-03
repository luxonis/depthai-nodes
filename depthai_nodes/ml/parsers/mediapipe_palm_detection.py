import cv2
import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .utils import generate_anchors_and_decode


class MPPalmDetectionParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the Mediapipe Palm detection model. As the
    result, the node sends out the detected hands in the form of a message containing
    bounding boxes, labels, and confidence scores.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.Parser sends the processed network results to this output in form of messages. It is a linking point from which the processed network results are retrieved.
    conf_threshold : float
        Confidence score threshold for detected hands.
    iou_threshold : float
        Non-maximum suppression threshold.
    max_det : int
        Maximum number of detections to keep.
    scale : int
        Scale of the input image.

    Output Message/s
    -------
    **Type**: dai.ImgDetections

    **Description**: ImgDetections message containing bounding boxes, labels, and confidence scores of detected hands.

    See also
    --------
    Official MediaPipe Hands solution:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    """

    def __init__(self, conf_threshold=0.5, iou_threshold=0.5, max_det=100, scale=192):
        """Initializes the MPPalmDetectionParser node.

        @param conf_threshold: Confidence score threshold for detected hands.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.scale = scale

    def setConfidenceThreshold(self, threshold):
        """Sets the confidence score threshold for detected hands.

        @param threshold: Confidence score threshold for detected hands.
        @type threshold: float
        """
        self.conf_threshold = threshold

    def setIOUThreshold(self, threshold):
        """Sets the non-maximum suppression threshold.

        @param threshold: Non-maximum suppression threshold.
        @type threshold: float
        """
        self.iou_threshold = threshold

    def setMaxDetections(self, max_det):
        """Sets the maximum number of detections to keep.

        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        """
        self.max_det = max_det

    def setScale(self, scale):
        """Sets the scale of the input image.

        @param scale: Scale of the input image.
        @type scale: int
        """
        self.scale = scale

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            all_tensors = output.getAllLayerNames()

            bboxes = None
            scores = None

            for tensor_name in all_tensors:
                tensor = output.getTensor(tensor_name, dequantize=True).astype(
                    np.float32
                )
                if bboxes is None:
                    bboxes = tensor
                    scores = tensor
                else:
                    bboxes = bboxes if tensor.shape[-1] < bboxes.shape[-1] else tensor
                    scores = tensor if tensor.shape[-1] < scores.shape[-1] else scores

            bboxes = bboxes.reshape(-1, 18)
            scores = scores.reshape(-1)

            if bboxes is None or scores is None:
                raise ValueError("No valid output tensors found.")

            decoded_bboxes = generate_anchors_and_decode(
                bboxes=bboxes,
                scores=scores,
                threshold=self.conf_threshold,
                scale=self.scale,
            )

            bboxes = []
            scores = []

            for hand in decoded_bboxes:
                extended_points = hand.rect_points
                xmin = int(min([point[0] for point in extended_points]))
                ymin = int(min([point[1] for point in extended_points]))
                xmax = int(max([point[0] for point in extended_points]))
                ymax = int(max([point[1] for point in extended_points]))

                bboxes.append([xmin, ymin, xmax, ymax])
                scores.append(hand.pd_score)

            indices = cv2.dnn.NMSBoxes(
                bboxes,
                scores,
                self.conf_threshold,
                self.iou_threshold,
                top_k=self.max_det,
            )
            bboxes = np.array(bboxes)[indices]
            scores = np.array(scores)[indices]

            bboxes = bboxes.astype(np.float32) / self.scale

            detections_msg = create_detection_message(bboxes, scores, labels=None)
            detections_msg.setTimestamp(output.getTimestamp())
            self.out.send(detections_msg)

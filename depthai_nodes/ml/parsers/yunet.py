import math

import cv2
import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .utils import decode_detections


class YuNetParser(dai.node.ThreadedHostNode):
    """YuNetParser class for parsing the output of the YuNet face detection model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in form of messages. It is a linking point from which the processed network results are retrieved.
    score_threshold : float
        Confidence score threshold for detected faces.
    nms_threshold : float
        Non-maximum suppression threshold.
    top_k : int
        Maximum number of detections to keep.

    Output Message/s
    ----------------
    **Type**: ImgDetectionsWithKeypoints

    **Description**: Message containing bounding boxes, labels, confidence scores, and keypoints of detected faces.
    """

    def __init__(
        self,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    ):
        """Initializes the YuNetParser node.

        @param score_threshold: Confidence score threshold for detected faces.
        @type score_threshold: float
        @param nms_threshold: Non-maximum suppression threshold.
        @type nms_threshold: float
        @param top_k: Maximum number of detections to keep.
        @type top_k: int
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def setConfidenceThreshold(self, threshold):
        """Sets the confidence score threshold for detected faces.

        @param threshold: Confidence score threshold for detected faces.
        @type threshold: float
        """
        self.score_threshold = threshold

    def setNMSThreshold(self, threshold):
        """Sets the non-maximum suppression threshold.

        @param threshold: Non-maximum suppression threshold.
        @type threshold: float
        """
        self.nms_threshold = threshold

    def setTopK(self, top_k):
        """Sets the maximum number of detections to keep.

        @param top_k: Maximum number of detections to keep.
        @type top_k: int
        """
        self.top_k = top_k

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            # get strides
            strides = list(
                set(
                    [
                        int(layer_name.split("_")[1])
                        for layer_name in output.getAllLayerNames()
                        if layer_name.startswith(("cls", "obj", "bbox", "kps"))
                    ]
                )
            )

            # get input_size
            stride0 = strides[0]
            _, spatial_positions0, _ = output.getTensor(f"cls_{stride0}").shape
            input_width = input_height = int(
                math.sqrt(spatial_positions0) * stride0
            )  # TODO: We assume a square input size. How to get input size when height and width are not equal?
            input_size = (input_width, input_height)

            detections = []
            for stride in strides:
                cls = output.getTensor(f"cls_{stride}").squeeze(0)
                obj = output.getTensor(f"obj_{stride}").flatten()
                bbox = output.getTensor(f"bbox_{stride}").squeeze(0)
                kps = output.getTensor(f"kps_{stride}").squeeze(0)
                detections += decode_detections(
                    input_size,
                    stride,
                    self.score_threshold,
                    cls,
                    obj,
                    bbox,
                    kps,
                )

            # non-maximum suppression
            detection_boxes = [detection["bbox"] for detection in detections]
            detection_scores = [detection["score"] for detection in detections]
            indices = cv2.dnn.NMSBoxes(
                detection_boxes,
                detection_scores,
                self.score_threshold,
                self.nms_threshold,
                top_k=self.top_k,
            )
            detections = np.array(detections)[indices]

            bboxes = []
            for detection in detections:
                xmin, ymin, width, height = detection["bbox"]
                bboxes.append([xmin, ymin, xmin + width, ymin + height])
            scores = [detection["score"] for detection in detections]
            labels = [detection["label"] for detection in detections]
            keypoints = [detection["keypoints"] for detection in detections]

            detections_message = create_detection_message(
                np.array(bboxes),
                np.array(scores),
                labels,
                keypoints,
            )

            self.out.send(detections_message)

import depthai as dai
import numpy as np
import cv2

from .utils import decode_detections
from .utils.message_creation import create_detections_msg


class YuNetParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        input_size=(640, 640),  # WH
        strides=[8, 16, 32],
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.input_size = input_size
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def setConfidenceThreshold(self, threshold):
        self.score_threshold = threshold

    def setNMSThreshold(self, threshold):
        self.nms_threshold = threshold

    def setTopK(self, top_k):
        self.top_k = top_k

    def setInputSize(self, width, height):
        self.input_size = (width, height)

    def setStrides(self, strides):
        self.strides = strides

    def run(self):
        """
        Postprocessing logic for YuNet model.

        Returns:
            dai.ImgDetectionsWithKeypoints: Detections with keypoints.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            detections = []
            for stride in self.strides:
                cols = int(self.input_size[1] / stride)  # w/stride
                rows = int(self.input_size[0] / stride)  # h/stride
                cls = output.getTensor(f"cls_{stride}").flatten()
                cls = np.expand_dims(
                    cls, axis=-1
                )  # add empty classes dimension (one class only in this case)

                obj = output.getTensor(f"obj_{stride}").flatten()
                bbox = output.getTensor(f"bbox_{stride}").squeeze(0)
                kps = output.getTensor(f"kps_{stride}").squeeze(0)

                detections += decode_detections(
                    self.input_size,
                    stride,
                    rows,
                    cols,
                    self.score_threshold,
                    cls,
                    obj,
                    bbox,
                    kps,
                )

            # non-maximum suppression
            if len(detections) > 1:
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


            detections_message = create_detections_msg(
                detections=detections,
                include_keypoints=True,
            )

            self.out.send(detections_message)

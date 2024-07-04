import cv2
import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .utils import generate_anchors_and_decode


class MPHandDetectionParser(dai.node.ThreadedHostNode):
    def __init__(self, score_threshold=0.5, nms_threshold=0.5, top_k=100):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def setConfidenceThreshold(self, threshold):
        self.score_threshold = threshold

    def setNMSThreshold(self, threshold):
        self.nms_threshold = threshold

    def setTopK(self, top_k):
        self.top_k = top_k

    def run(self):
        """Postprocessing logic for MediPipe Hand detection model.

        Returns:
            dai.ImgDetections containing bounding boxes, labels, and confidence scores of detected hands.
        """

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            bboxes = output.getTensor("Identity").reshape(2016, 18).astype(np.float32)
            scores = output.getTensor("Identity_1").reshape(2016).astype(np.float32)

            decoded_bboxes = generate_anchors_and_decode(
                bboxes=bboxes, scores=scores, threshold=self.score_threshold, scale=192
            )

            bboxes = []
            scores = []

            for hand in decoded_bboxes:
                extended_points = hand.rect_points
                xmin = int(min(extended_points[0][0], extended_points[1][0]))
                ymin = int(min(extended_points[0][1], extended_points[1][1]))
                xmax = int(max(extended_points[2][0], extended_points[3][0]))
                ymax = int(max(extended_points[2][1], extended_points[3][1]))

                bboxes.append([xmin, ymin, xmax, ymax])
                scores.append(hand.pd_score)

            indices = cv2.dnn.NMSBoxes(
                bboxes,
                scores,
                self.score_threshold,
                self.nms_threshold,
                top_k=self.top_k,
            )
            bboxes = np.array(bboxes)[indices]
            scores = np.array(scores)[indices]

            detections_msg = create_detection_message(bboxes, scores, labels=None)
            self.out.send(detections_msg)

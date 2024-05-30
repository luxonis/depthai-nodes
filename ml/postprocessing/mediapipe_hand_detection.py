import depthai as dai
import numpy as np
import cv2

from .utils.medipipe import generate_handtracker_anchors, decode_bboxes, rect_transformation, detections_to_rect

class MPHandDetectionParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        score_threshold=0.5,
        nms_threshold=0.5,
        top_k=100
    ):
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
        """
        Postprocessing logic for MediPipe Hand detection model.

        Returns:
            dai.ImgDetections containing bounding boxes, labels, and confidence scores of detected hands.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            tensorInfo = output.getTensorInfo("Identity")
            bboxes = output.getTensor(f"Identity").reshape(2016, 18).astype(np.float32)
            bboxes = (bboxes - tensorInfo.qpZp) * tensorInfo.qpScale
            tensorInfo = output.getTensorInfo("Identity_1")
            scores = output.getTensor(f"Identity_1").reshape(2016).astype(np.float32)
            scores = (scores - tensorInfo.qpZp) * tensorInfo.qpScale

            anchors = generate_handtracker_anchors(192, 192)
            decoded_bboxes = decode_bboxes(0.5, scores, bboxes, anchors, scale=192)
            detections_to_rect(decoded_bboxes)
            rect_transformation(decoded_bboxes, 192, 192)
            
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
            
            indices = cv2.dnn.NMSBoxes(bboxes, scores, self.score_threshold, self.nms_threshold, top_k=self.top_k)
            bboxes = np.array(bboxes)[indices]
            scores = np.array(scores)[indices]

            detections = []
            for bbox, score in zip(bboxes, scores):
                detection = dai.ImgDetection()
                detection.confidence = score
                detection.label = 0
                detection.xmin = bbox[0]
                detection.ymin = bbox[1]
                detection.xmax = bbox[2]
                detection.ymax = bbox[3]
                detections.append(detection)

            detections_msg = dai.ImgDetections()
            detections_msg.detections = detections

            self.out.send(detections_msg)
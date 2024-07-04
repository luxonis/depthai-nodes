import math

import cv2
import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .utils import decode_detections


class YuNetParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
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
        """Postprocessing logic for YuNet model.

        Returns:
            dai.ImgDetectionsWithKeypoints: Detections with keypoints.
        """

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
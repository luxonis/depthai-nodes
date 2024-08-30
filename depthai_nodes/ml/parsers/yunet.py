import math

import cv2
import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .utils import decode_detections


class YuNetParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the YuNet face detection model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    conf_threshold : float
        Confidence score threshold for detected faces.
    iou_threshold : float
        Non-maximum suppression threshold.
    max_det : int
        Maximum number of detections to keep.

    Output Message/s
    ----------------
    **Type**: ImgDetectionsExtended

    **Description**: Message containing bounding boxes, labels, confidence scores, and keypoints of detected faces.
    """

    def __init__(
        self,
        conf_threshold=0.6,
        iou_threshold=0.3,
        max_det=5000,
    ):
        """Initializes the YuNetParser node.

        @param conf_threshold: Confidence score threshold for detected faces.
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

    def setConfidenceThreshold(self, threshold):
        """Sets the confidence score threshold for detected faces.

        @param threshold: Confidence score threshold for detected faces.
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
            cls_stride0_shape = output.getTensor(
                f"cls_{stride0}", dequantize=True
            ).shape
            if len(cls_stride0_shape) == 3:
                _, spatial_positions0, _ = cls_stride0_shape
            elif len(cls_stride0_shape) == 2:
                spatial_positions0, _ = cls_stride0_shape
            input_width = input_height = int(
                math.sqrt(spatial_positions0) * stride0
            )  # TODO: We assume a square input size. How to get input size when height and width are not equal?
            input_size = (input_width, input_height)

            detections = []
            for stride in strides:
                cls = output.getTensor(f"cls_{stride}", dequantize=True)
                cls = cls.astype(np.float32)
                cls = cls.squeeze(0) if cls.shape[0] == 1 else cls

                obj = output.getTensor(f"obj_{stride}", dequantize=True).flatten()
                obj = obj.astype(np.float32)

                bbox = output.getTensor(f"bbox_{stride}", dequantize=True)
                bbox = bbox.astype(np.float32)
                bbox = bbox.squeeze(0) if bbox.shape[0] == 1 else bbox

                kps = output.getTensor(f"kps_{stride}", dequantize=True)
                kps = kps.astype(np.float32)
                kps = kps.squeeze(0) if kps.shape[0] == 1 else kps

                detections += decode_detections(
                    input_size,
                    stride,
                    self.conf_threshold,
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
                self.conf_threshold,
                self.iou_threshold,
                top_k=self.max_det,
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
            detections_message.setTimestamp(output.getTimestamp())

            self.out.send(detections_message)

import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .utils.scrfd import decode_scrfd


class SCRFDParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the SCRFD face detection model.

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
    feat_stride_fpn : tuple
        Tuple of the feature strides.
    num_anchors : int
        Number of anchors.
    input_size : tuple
        Input size of the model.

    Output Message/s
    ----------------
    **Type**: dai.ImgDetections

    **Description**: ImgDetections message containing bounding boxes, labels, and confidence scores of detected faces.
    """

    def __init__(
        self,
        conf_threshold=0.5,
        iou_threshold=0.5,
        max_det=100,
        input_size=(640, 640),
        feat_stride_fpn=(8, 16, 32),
        num_anchors=2,
    ):
        """Initializes the SCRFDParser node.

        @param conf_threshold: Confidence score threshold for detected faces.
        @type conf_threshold: float
        @param iou_threshold: Non-maximum suppression threshold.
        @type iou_threshold: float
        @param max_det: Maximum number of detections to keep.
        @type max_det: int
        @param feat_stride_fpn: List of the feature strides.
        @type feat_stride_fpn: tuple
        @param num_anchors: Number of anchors.
        @type num_anchors: int
        @param input_size: Input size of the model.
        @type input_size: tuple
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        self.feat_stride_fpn = feat_stride_fpn
        self.num_anchors = num_anchors
        self.input_size = input_size

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

    def setFeatStrideFPN(self, feat_stride_fpn):
        """Sets the feature stride of the FPN.

        @param feat_stride_fpn: Feature stride of the FPN.
        @type feat_stride_fpn: list
        """
        self.feat_stride_fpn = feat_stride_fpn

    def setInputSize(self, input_size):
        """Sets the input size of the model.

        @param input_size: Input size of the model.
        @type input_size: list
        """
        self.input_size = input_size

    def setNumAnchors(self, num_anchors):
        """Sets the number of anchors.

        @param num_anchors: Number of anchors.
        @type num_anchors: int
        """
        self.num_anchors = num_anchors

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            scores_concatenated = []
            bboxes_concatenated = []
            kps_concatenated = []

            for stride in self.feat_stride_fpn:
                score_layer_name = f"score_{stride}"
                bbox_layer_name = f"bbox_{stride}"
                kps_layer_name = f"kps_{stride}"
                if score_layer_name not in output.getAllLayerNames():
                    raise ValueError(
                        f"Layer {score_layer_name} not found in the model output."
                    )
                if bbox_layer_name not in output.getAllLayerNames():
                    raise ValueError(
                        f"Layer {bbox_layer_name} not found in the model output."
                    )
                if kps_layer_name not in output.getAllLayerNames():
                    raise ValueError(
                        f"Layer {kps_layer_name} not found in the model output."
                    )

                score_tensor = (
                    output.getTensor(score_layer_name, dequantize=True)
                    .flatten()
                    .astype(np.float32)
                )
                bbox_tensor = (
                    output.getTensor(bbox_layer_name, dequantize=True)
                    .reshape(len(score_tensor), 4)
                    .astype(np.float32)
                )
                kps_tensor = (
                    output.getTensor(kps_layer_name, dequantize=True)
                    .reshape(len(score_tensor), 10)
                    .astype(np.float32)
                )

                scores_concatenated.append(score_tensor)
                bboxes_concatenated.append(bbox_tensor)
                kps_concatenated.append(kps_tensor)

            bboxes, scores, keypoints = decode_scrfd(
                bboxes_concatenated=bboxes_concatenated,
                scores_concatenated=scores_concatenated,
                kps_concatenated=kps_concatenated,
                feat_stride_fpn=self.feat_stride_fpn,
                input_size=self.input_size,
                num_anchors=self.num_anchors,
                score_threshold=self.conf_threshold,
                nms_threshold=self.iou_threshold,
            )
            detection_msg = create_detection_message(
                bboxes, scores, None, keypoints.tolist()
            )
            detection_msg.setTimestamp(output.getTimestamp())

            self.out.send(detection_msg)

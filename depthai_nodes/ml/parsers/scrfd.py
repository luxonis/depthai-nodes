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
    score_threshold : float
        Confidence score threshold for detected faces.
    nms_threshold : float
        Non-maximum suppression threshold.
    top_k : int
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
        score_threshold=0.5,
        nms_threshold=0.5,
        top_k=100,
        input_size=(640, 640),
        feat_stride_fpn=(8, 16, 32),
        num_anchors=2,
    ):
        """Initializes the SCRFDParser node.

        @param score_threshold: Confidence score threshold for detected faces.
        @type score_threshold: float
        @param nms_threshold: Non-maximum suppression threshold.
        @type nms_threshold: float
        @param top_k: Maximum number of detections to keep.
        @type top_k: int
        @param feat_stride_fpn: List of the feature strides.
        @type feat_stride_fpn: tuple
        @param num_anchors: Number of anchors.
        @type num_anchors: int
        @param input_size: Input size of the model.
        @type input_size: tuple
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        self.feat_stride_fpn = feat_stride_fpn
        self.num_anchors = num_anchors
        self.input_size = input_size

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

            score_8 = (
                output.getTensor("score_8", dequantize=True)
                .flatten()
                .astype(np.float32)
            )
            score_16 = (
                output.getTensor("score_16", dequantize=True)
                .flatten()
                .astype(np.float32)
            )
            score_32 = (
                output.getTensor("score_32", dequantize=True)
                .flatten()
                .astype(np.float32)
            )
            bbox_8 = (
                output.getTensor("bbox_8", dequantize=True)
                .reshape(len(score_8), 4)
                .astype(np.float32)
            )
            bbox_16 = (
                output.getTensor("bbox_16", dequantize=True)
                .reshape(len(score_16), 4)
                .astype(np.float32)
            )
            bbox_32 = (
                output.getTensor("bbox_32", dequantize=True)
                .reshape(len(score_32), 4)
                .astype(np.float32)
            )
            kps_8 = (
                output.getTensor("kps_8", dequantize=True)
                .reshape(len(score_8), 10)
                .astype(np.float32)
            )
            kps_16 = (
                output.getTensor("kps_16", dequantize=True)
                .reshape(len(score_16), 10)
                .astype(np.float32)
            )
            kps_32 = (
                output.getTensor("kps_32", dequantize=True)
                .reshape(len(score_32), 10)
                .astype(np.float32)
            )

            bboxes_concatenated = [bbox_8, bbox_16, bbox_32]
            scores_concatenated = [score_8, score_16, score_32]
            kps_concatenated = [kps_8, kps_16, kps_32]

            bboxes, scores, keypoints = decode_scrfd(
                bboxes_concatenated=bboxes_concatenated,
                scores_concatenated=scores_concatenated,
                kps_concatenated=kps_concatenated,
                feat_stride_fpn=self.feat_stride_fpn,
                input_size=self.input_size,
                num_anchors=self.num_anchors,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
            )
            detection_msg = create_detection_message(
                bboxes, scores, None, keypoints.tolist()
            )

            bboxes = np.array(bboxes)[indices]
            keypoints = np.array(keypoints)[indices]
            scores = scores[indices]

            detection_msg = create_detection_message(bboxes, scores, None, None)
            detection_msg.setTimestamp(output.getTimestamp())

            self.out.send(detection_msg)

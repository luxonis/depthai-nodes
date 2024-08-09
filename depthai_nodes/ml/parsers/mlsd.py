import depthai as dai
import numpy as np

from ..messages.creators import create_line_detection_message
from .utils.mlsd import decode_scores_and_points, get_lines


class MLSDParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the M-LSD line detection model. The parser
    is specifically designed to parse the output of the M-LSD model. As the result, the
    node sends out the detected lines in the form of a message.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    nn_passthrough : Node.Input
        Node's 2nd input. It accepts the passthrough of the Neural Network node. This is required for parsing the output of the M-LSD model.
        It is a linking point to which the Neural Network's passthrough (network's input accutualy) is linked.
    topk_n : int
        Number of top candidates to keep.
    score_thr : float
        Confidence score threshold for detected lines.
    dist_thr : float
        Distance threshold for merging lines.

    Output Message/s
    ----------------
    **Type**: LineDetections

    **Description**: LineDetections message containing detected lines and confidence scores.
    """

    def __init__(
        self,
        topk_n=200,
        score_thr=0.10,
        dist_thr=20.0,
    ):
        """Initializes the MLSDParser node.

        @param topk_n: Number of top candidates to keep.
        @type topk_n: int
        @param score_thr: Confidence score threshold for detected lines.
        @type score_thr: float
        @param dist_thr: Distance threshold for merging lines.
        @type dist_thr: float
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.nn_passthrough = dai.Node.Input(self)
        self.out = dai.Node.Output(self)
        self.topk_n = topk_n
        self.score_thr = score_thr
        self.dist_thr = dist_thr

    def setTopK(self, topk_n):
        """Sets the number of top candidates to keep.

        @param topk_n: Number of top candidates to keep.
        @type topk_n: int
        """
        self.topk_n = topk_n

    def setScoreThreshold(self, score_thr):
        """Sets the confidence score threshold for detected lines.

        @param score_thr: Confidence score threshold for detected lines.
        @type score_thr: float
        """
        self.score_thr = score_thr

    def setDistanceThreshold(self, dist_thr):
        """Sets the distance threshold for merging lines.

        @param dist_thr: Distance threshold for merging lines.
        @type dist_thr: float
        """
        self.dist_thr = dist_thr

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
                nn_passthrough: dai.NNData = self.nn_passthrough.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            tpMap = nn_passthrough.getTensor("output").astype(np.float32)
            heat_np = output.getTensor("heat").astype(np.float32)

            pts, pts_score, vmap = decode_scores_and_points(tpMap, heat_np, self.topk_n)
            lines, scores = get_lines(
                pts, pts_score, vmap, self.score_thr, self.dist_thr
            )

            message = create_line_detection_message(lines, np.array(scores))
            message.setTimestamp(output.getTimestamp())
            self.out.send(message)

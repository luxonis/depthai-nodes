import depthai as dai
import numpy as np

from .utils.mlsd import decode_scores_and_points, get_lines
from ..messages.creators import create_line_detection_message

class MLSDParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        topk_n=200,
        score_thr=0.10,
        dist_thr=20.0,
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.nn_passthrough = dai.Node.Input(self)
        self.out = dai.Node.Output(self)
        self.topk_n = topk_n
        self.score_thr = score_thr
        self.dist_thr = dist_thr

    def setTopK(self, topk_n):
        self.topk_n = topk_n
    
    def setScoreThreshold(self, score_thr):
        self.score_thr = score_thr
    
    def setDistanceThreshold(self, dist_thr):
        self.dist_thr = dist_thr

    def run(self):
        """
        Postprocessing logic for M-LSD line detection model.

        Returns:
            Normalized detected lines and confidence scores.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
                nn_passthrough: dai.NNData = self.nn_passthrough.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            tpMap = nn_passthrough.getTensor("output").astype(np.float32)
            heat_np = output.getTensor("heat").astype(np.float32)

            pts, pts_score, vmap = decode_scores_and_points(tpMap, heat_np, self.topk_n)
            lines, scores = get_lines(pts, pts_score, vmap, self.score_thr, self.dist_thr)

            message = create_line_detection_message(lines, np.array(scores))
            self.out.send(message)
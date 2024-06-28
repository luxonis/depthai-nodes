import depthai as dai
import numpy as np
from typing import Tuple
from .utils.xfeat import detect_and_compute, match
from ..messages.creators import create_tracked_features_message

class XFeatParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        original_size: Tuple[float, float] = None,
        input_size: Tuple[float, float] = (640, 352)
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)
        self.original_size = original_size
        self.input_size = input_size
        self.previous_results = None

    def setOriginalSize(self, original_size):
        self.original_size = original_size
    
    def setInputSize(self, input_size):
        self.input_size = input_size

    def run(self):
        """
        Postprocessing logic for XFeat model.

        Returns:
            dai.MatchedPoints containing matched keypoints.
        """
        if self.original_size is None:
            raise ValueError("Original image size must be specified!")

        resize_rate_w = self.original_size[0] / self.input_size[0]
        resize_rate_h = self.original_size[1] / self.input_size[1]

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped


            feats = output.getTensor(f"feats").astype(np.float32)
            heatmaps = output.getTensor(f"heatmaps").astype(np.float32)
            keypoints = output.getTensor(f"keypoints").astype(np.float32)

            result = detect_and_compute(feats, keypoints, resize_rate_w, resize_rate_h, self.input_size)[0]

            if self.previous_results is not None:
                mkpts0, mkpts1 = match(self.previous_results, result)
                matched_points = create_tracked_features_message(mkpts0, mkpts1)
                self.out.send(matched_points)
            else:
                # save the result from first frame
                self.previous_results = result
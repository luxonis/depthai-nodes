import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message
from .utils.superanimal import get_pose_prediction

class SuperAnimalParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        score_threshold=0.5,
        scale_factor=256,
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.score_threshold = score_threshold
        self.scale_factor = scale_factor

    def setScoreThreshold(self, threshold):
        self.score_threshold = threshold

    def setScaleFactor(self, scale_factor):
        self.scale_factor = scale_factor

    def run(self):
        """
        Postprocessing logic for SuperAnimal landmark model.

        Returns:
            dai.Keypoints: Max 39 keypoints detected on the quadrupedal animal.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            heatmaps = output.getTensor(f"heatmaps").astype(np.float32)

            heatmaps_scale_factor = (self.scale_factor / heatmaps.shape[1], self.scale_factor / heatmaps.shape[2])

            keypoints = get_pose_prediction(heatmaps, None, heatmaps_scale_factor)[0][0]
            scores = keypoints[:, 2]
            keypoints = keypoints[:, :2] / self.scale_factor

            msg = create_keypoints_message(keypoints, scores, self.score_threshold)

            self.out.send(msg)
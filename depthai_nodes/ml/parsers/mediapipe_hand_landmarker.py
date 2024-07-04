import depthai as dai
import numpy as np

from ..messages.creators import create_hand_keypoints_message


class MPHandLandmarkParser(dai.node.ThreadedHostNode):
    def __init__(self, score_threshold=0.5, scale_factor=224):
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
        """Postprocessing logic for MediaPipe Hand landmark model.

        Returns:
            HandLandmarks containing normalized 21 landmarks, confidence score, and handdedness score (right or left hand).
        """

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            landmarks = output.getTensor("Identity").reshape(21, 3).astype(np.float32)
            hand_score = output.getTensor("Identity_1").reshape(-1).astype(np.float32)
            handedness = output.getTensor("Identity_2").reshape(-1).astype(np.float32)
            hand_score = hand_score[0]
            handedness = handedness[0]

            # normalize landmarks
            landmarks /= self.scale_factor

            hand_landmarks_msg = create_hand_keypoints_message(
                landmarks, float(handedness), float(hand_score), self.score_threshold
            )
            self.out.send(hand_landmarks_msg)

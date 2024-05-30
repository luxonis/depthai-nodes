import depthai as dai
import numpy as np
import cv2

from ..messages import HandKeypoints

class MPHandLandmarkParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        score_threshold=0.5
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.score_threshold = score_threshold

    def setScoreThreshold(self, threshold):
        self.score_threshold = threshold

    def run(self):
        """
        Postprocessing logic for MediaPipe Hand landmark model.

        Returns:
            HandLandmarks containing 21 landmarks, confidence score, and handdedness score (right or left hand).
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            tensorInfo = output.getTensorInfo("Identity")
            landmarks = output.getTensor(f"Identity").reshape(21, 3).astype(np.float32)
            landmarks = (landmarks - tensorInfo.qpZp) * tensorInfo.qpScale
            tensorInfo = output.getTensorInfo("Identity_1")
            hand_score = output.getTensor(f"Identity_1").reshape(-1).astype(np.float32)
            hand_score = (hand_score - tensorInfo.qpZp) * tensorInfo.qpScale
            hand_score = hand_score[0]
            tensorInfo = output.getTensorInfo("Identity_2")
            handdedness = output.getTensor(f"Identity_2").reshape(-1).astype(np.float32)
            handdedness = (handdedness - tensorInfo.qpZp) * tensorInfo.qpScale
            handdedness = handdedness[0]

            hand_landmarks_msg = HandKeypoints()
            hand_landmarks_msg.handdedness = handdedness
            hand_landmarks_msg.confidence = hand_score
            hand_landmarks = []
            if hand_score >= self.score_threshold:
                for i in range(21):
                    pt = dai.Point3f()
                    pt.x = landmarks[i][0]
                    pt.y = landmarks[i][1]
                    pt.z = landmarks[i][2]
                    hand_landmarks.append(pt)
            hand_landmarks_msg.landmarks = hand_landmarks
            self.out.send(hand_landmarks_msg)
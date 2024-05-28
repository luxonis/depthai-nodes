import depthai as dai
import numpy as np
import cv2

from ..messages import HandLandmarks

class MPHandLandmarkParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        score_threshold=0.5,
        handdedness_threshold=0.5,
        input_size=(224, 224)
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.score_threshold = score_threshold
        self.input_size = input_size
        self.handdedness_threshold = handdedness_threshold

    def setScoreThreshold(self, threshold):
        self.score_threshold = threshold

    def setHandednessThreshold(self, threshold):
        self.handdedness_threshold = threshold

    def setInputSize(self, width, height):
        self.input_size = (width, height)

    def run(self):
        """
        Postprocessing logic for SCRFD model.

        Returns:
            ...
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            print('MP Hand landmark node')
            print(f"Layer names = {output.getAllLayerNames()}")

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

            hand_landmarks_msg = HandLandmarks()
            if hand_score < self.score_threshold:
                hand_landmarks_msg.landmarks = []
                hand_landmarks_msg.confidence = hand_score
                hand_landmarks_msg.handdedness = handdedness
                self.out.send(hand_landmarks_msg)
            else:
                hand_landmarks_msg.confidence = hand_score
                hand_landmarks_msg.handdedness = handdedness
                for i in range(21):
                    pt = dai.Point3f()
                    pt.x = landmarks[i][0]
                    pt.y = landmarks[i][1]
                    pt.z = landmarks[i][2]
                    hand_landmarks_msg.landmarks.append(pt)
                self.out.send(hand_landmarks_msg)
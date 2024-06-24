import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message

class MPFaceLandmarkerParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        scale_factor=192,
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.scale_factor = scale_factor

    def setScaleFactor(self, scale_factor):
        self.scale_factor = scale_factor

    def run(self):
        """
        Postprocessing logic for Mediapipe face mesh model.

        Returns:
            dai.Keypoints: 468 3D keypoints detected on the face.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            tensorInfo = output.getTensorInfo("conv2d_21_1")
            landmarks = output.getTensor("conv2d_21_1").reshape(468, 3).astype(np.float32)
            landmarks = (landmarks - tensorInfo.qpZp) * tensorInfo.qpScale

            landmarks /= self.scale_factor

            msg = create_keypoints_message(landmarks)

            self.out.send(msg)

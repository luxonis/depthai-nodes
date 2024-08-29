import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message
from .utils.superanimal import get_pose_prediction


class SuperAnimalParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the SuperAnimal landmark model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    score_threshold : float
        Confidence score threshold for detected keypoints.
    scale_factor : float
        Scale factor to divide the keypoints by.

    Output Message/s
    ----------------
    **Type**: Keypoints

    **Description**: Keypoints message containing detected keypoints that exceeds confidence threshold.
    """

    def __init__(
        self,
        score_threshold=0.5,
        scale_factor=256,
    ):
        """Initializes the SuperAnimalParser node.

        @param score_threshold: Confidence score threshold for detected keypoints.
        @type score_threshold: float
        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.score_threshold = score_threshold
        self.scale_factor = scale_factor

    def setScoreThreshold(self, threshold):
        """Sets the confidence score threshold for detected keypoints.

        @param threshold: Confidence score threshold for detected keypoints.
        @type threshold: float
        """
        self.score_threshold = threshold

    def setScaleFactor(self, scale_factor):
        """Sets the scale factor to divide the keypoints by.

        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        """
        self.scale_factor = scale_factor

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            heatmaps = output.getTensor("heatmaps", dequantize=True).astype(np.float32)

            if len(heatmaps.shape) == 3:
                heatmaps = heatmaps.reshape((1,) + heatmaps.shape)

            heatmaps_scale_factor = (
                self.scale_factor / heatmaps.shape[1],
                self.scale_factor / heatmaps.shape[2],
            )

            keypoints = get_pose_prediction(heatmaps, None, heatmaps_scale_factor)[0][0]
            scores = keypoints[:, 2]
            keypoints = keypoints[:, :2] / self.scale_factor

            msg = create_keypoints_message(keypoints, scores, self.score_threshold)
            msg.setTimestamp(output.getTimestamp())

            self.out.send(msg)

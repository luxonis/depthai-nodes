import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message


class HRNetParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the HRNet pose estimation model. The code is inspired by https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    score_threshold : float
        Confidence score threshold for detected keypoints.

    Output Message/s
    ----------------
    **Type**: Keypoints

    **Description**: Keypoints message containing detected body keypoints.
    """

    def __init__(self, score_threshold=0.5):
        """Initializes the HRNetParser node.

        @param score_threshold: Confidence score threshold for detected keypoints.
        @type score_threshold: float
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.score_threshold = score_threshold

    def setScoreThreshold(self, threshold):
        """Sets the confidence score threshold for the detected body keypoints.

        @param threshold: Confidence score threshold for detected keypoints.
        @type threshold: float
        """
        self.score_threshold = threshold

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            heatmaps = output.getTensor("heatmaps", dequantize=True)

            if len(heatmaps.shape) == 4:
                heatmaps = heatmaps[0]
            if heatmaps.shape[2] == 16:  # HW_ instead of _HW
                heatmaps = heatmaps.transpose(2, 0, 1)
            _, map_h, map_w = heatmaps.shape

            scores = np.array([np.max(heatmap) for heatmap in heatmaps])
            keypoints = np.array(
                [
                    np.unravel_index(heatmap.argmax(), heatmap.shape)
                    for heatmap in heatmaps
                ]
            )
            keypoints = keypoints.astype(np.float32)
            keypoints = keypoints[:, ::-1] / np.array(
                [map_w, map_h]
            )  # normalize keypoints to [0, 1]

            keypoints_message = create_keypoints_message(
                keypoints=keypoints,
                scores=scores,
                confidence_threshold=self.score_threshold,
            )
            keypoints_message.setTimestamp(output.getTimestamp())

            self.out.send(keypoints_message)

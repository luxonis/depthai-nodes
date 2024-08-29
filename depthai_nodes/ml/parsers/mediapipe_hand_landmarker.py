import depthai as dai
import numpy as np

from ..messages.creators import create_hand_keypoints_message


class MPHandLandmarkParser(dai.node.ThreadedHostNode):
    """Parser class for MediaPipe Hand landmark model. It parses the output of the
    MediaPipe Hand landmark model containing 21 3D hand landmarks. The landmarks are
    normalized and sent as a message to the output. Besides landmarks, the message
    contains confidence score and handedness score (right or left hand).

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    score_threshold : float
        Confidence score threshold for hand landmarks.
    scale_factor : float
        Scale factor to divide the landmarks by.

    Output Message/s
    ----------------
    **Type**: HandLandmarks

    **Description**: HandLandmarks message containing normalized 21 3D landmarks, confidence score, and handedness score (right or left hand).

    See also
    --------
    Official MediaPipe Hands solution:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    """

    def __init__(self, score_threshold=0.5, scale_factor=224):
        """Initialize MPHandLandmarkParser node.

        @param score_threshold: Confidence score threshold for hand landmarks.
        @type score_threshold: float
        @param scale_factor: Scale factor to divide the landmarks by.
        @type scale_factor: float
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.score_threshold = score_threshold
        self.scale_factor = scale_factor

    def setScoreThreshold(self, threshold):
        """Set the confidence score threshold for hand landmarks.

        @param threshold: Confidence score threshold for hand landmarks.
        @type threshold: float
        """
        self.score_threshold = threshold

    def setScaleFactor(self, scale_factor):
        """Set the scale factor to divide the landmarks by.

        @param scale_factor: Scale factor to divide the landmarks by.
        @type scale_factor: float
        """
        self.scale_factor = scale_factor

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            landmarks = (
                output.getTensor("Identity", dequantize=True)
                .reshape(21, 3)
                .astype(np.float32)
            )
            hand_score = (
                output.getTensor("Identity_1", dequantize=True)
                .reshape(-1)
                .astype(np.float32)
            )
            handedness = (
                output.getTensor("Identity_2", dequantize=True)
                .reshape(-1)
                .astype(np.float32)
            )
            hand_score = hand_score[0]
            handedness = handedness[0]

            # normalize landmarks
            landmarks /= self.scale_factor

            hand_landmarks_msg = create_hand_keypoints_message(
                landmarks, float(handedness), float(hand_score), self.score_threshold
            )
            hand_landmarks_msg.setTimestamp(output.getTimestamp())
            self.out.send(hand_landmarks_msg)

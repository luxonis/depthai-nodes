import depthai as dai
import numpy as np

from ..messages.creators import create_hand_keypoints_message, create_keypoints_message
from .keypoints import KeypointParser


class MPKeypointsParser(KeypointParser):
    """Parser class for keypoint models from MediaPipe. It expects one output layer containing keypoints and one output layer containing confidence score - whether the object is present or not.
    It also allows the third output layer for handedness score (right or left hand) - in case of using Hand Landmark model.

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
    n_keypoints : int
        Number of keypoints the model detects.

    Output Message/s
    ----------------
    **Type**: HandKeypoints or KeypointsWithConfidence

    **Description**: HandKeypoints message containing normalized 21 3D keypoints, confidence score, and handedness score (right or left hand) or KeypointsWithConfidence message containing normalized 9 keypoints and confidence score.

    """

    def __init__(self, score_threshold=0.5, scale_factor=224, n_keypoints=21):
        """Initialize MPHandLandmarkParser node.

        @param score_threshold: Confidence score threshold for hand landmarks.
        @type score_threshold: float
        @param scale_factor: Scale factor to divide the landmarks by.
        @type scale_factor: float
        @param n_keypoints: Number of keypoints the model detects.
        @type n_keypoints: int
        """
        KeypointParser.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.score_threshold = score_threshold
        self.scale_factor = scale_factor
        self.n_keypoints = n_keypoints

    def setScoreThreshold(self, threshold):
        """Set the confidence score threshold for hand landmarks.

        @param threshold: Confidence score threshold for hand landmarks.
        @type threshold: float
        """
        self.score_threshold = threshold

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()

            keypoints = output.getTensor("keypoints", dequantize=True).astype(
                np.float32
            )
            score = (
                output.getTensor("score", dequantize=True)
                .astype(np.float32)
                .reshape(-1)[0]
            )
            handedness = None
            if "handedness" in output_layer_names:
                handedness = (
                    output.getTensor(output_layer_names[2], dequantize=True)
                    .astype(np.float32)
                    .reshape(-1)[0]
                )

            num_coords = int(np.prod(keypoints.shape) / self.n_keypoints)

            if num_coords not in [2, 3]:
                raise ValueError(
                    f"Expected 2 or 3 coordinates per keypoint, got {num_coords}."
                )

            keypoints = keypoints.reshape(self.n_keypoints, num_coords)
            print(keypoints)
            # normalize keypoints
            keypoints /= self.scale_factor

            message = (
                create_keypoints_message(
                    keypoints=keypoints,
                    scores=None,
                    confidence_threshold=None,
                    objectness=float(score),
                    objectness_threshold=self.score_threshold,
                )
                if not handedness
                else create_hand_keypoints_message(
                    keypoints, float(handedness), float(score), self.score_threshold
                )
            )

            message.setTimestamp(output.getTimestamp())
            self.out.send(message)

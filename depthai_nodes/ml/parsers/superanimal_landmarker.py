from typing import Any, Dict

import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message
from .keypoints import KeypointParser
from .utils.superanimal import get_pose_prediction


class SuperAnimalParser(KeypointParser):
    """Parser class for parsing the output of the SuperAnimal landmark model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_layer_name: str
        Name of the output layer from which the scores are extracted.
    n_keypoints : int
        Number of keypoints.
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
        output_layer_name="",
        n_keypoints=39,
        score_threshold=0.5,
        scale_factor=256,
    ):
        """Initializes the SuperAnimalParser node.

        @param output_layer_name: Name of the output layer from which the keypoints are
            extracted.
        @type output_layer_name: str
        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        @param score_threshold: Confidence score threshold for detected keypoints.
        @type score_threshold: float
        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        """
        super().__init__(output_layer_name, scale_factor, n_keypoints=n_keypoints)
        self.output_layer_name = output_layer_name

        self.score_threshold = score_threshold

    def build(
        self,
        head_config: Dict[str, Any],
    ):
        """Sets the head configuration for the parser.

        Attributes
        ----------
        head_config : Dict
            The head configuration for the parser.

        Returns
        -------
        SuperAnimalParser
            Returns the parser object with the head configuration set.
        """

        super().build(head_config)
        self.score_threshold = head_config["score_threshold"]

        return self

    def setScoreThreshold(self, threshold):
        """Sets the confidence score threshold for detected keypoints.

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

            layers = output.getAllLayerNames()

            if len(layers) == 1 and self.output_layer_name == "":
                self.output_layer_name = layers[0]
            elif len(layers) != 1 and self.output_layer_name == "":
                raise ValueError(
                    f"Expected 1 output layer, got {len(layers)} layers. Please provide the output_layer_name."
                )
            heatmaps = output.getTensor(self.output_layer_name, dequantize=True).astype(
                np.float32
            )

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

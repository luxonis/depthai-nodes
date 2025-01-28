from typing import Any, Dict

import depthai as dai
import numpy as np

from depthai_nodes.ml.messages.creators import create_keypoints_message
from depthai_nodes.ml.parsers.keypoints import KeypointParser
from depthai_nodes.ml.parsers.utils.superanimal import get_pose_prediction


class SuperAnimalParser(KeypointParser):
    """Parser class for parsing the output of the SuperAnimal landmark model.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    scale_factor : float
        Scale factor to divide the keypoints by.
    n_keypoints : int
        Number of keypoints.
    score_threshold : float
        Confidence score threshold for detected keypoints.

    Output Message/s
    ----------------
    **Type**: Keypoints

    **Description**: Keypoints message containing detected keypoints that exceeds confidence threshold.
    """

    def __init__(
        self,
        output_layer_name: str = "",
        scale_factor: float = 256.0,
        n_keypoints: int = 39,
        score_threshold: float = 0.5,
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        @param score_threshold: Confidence score threshold for detected keypoints.
        @type score_threshold: float
        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        """
        super().__init__(
            output_layer_name,
            scale_factor=scale_factor,
            n_keypoints=n_keypoints,
            score_threshold=score_threshold,
        )

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "SuperAnimalParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: SuperAnimalParser
        """

        super().build(head_config)

        return self

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

            heatmaps_scale_factor = (
                self.scale_factor / heatmaps.shape[1],
                self.scale_factor / heatmaps.shape[2],
            )

            keypoints = get_pose_prediction(heatmaps, None, heatmaps_scale_factor)[0][0]
            scores = keypoints[:, 2]
            keypoints = keypoints[:, :2] / self.scale_factor

            msg = create_keypoints_message(keypoints, scores, self.score_threshold)
            msg.setTimestamp(output.getTimestamp())
            msg.transformation = output.getTransformation()
            msg.setSequenceNum(output.getSequenceNum())

            self.out.send(msg)

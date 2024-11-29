from typing import Any, Dict

import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message
from .keypoints import KeypointParser


class HRNetParser(KeypointParser):
    """Parser class for parsing the output of the HRNet pose estimation model. The code is inspired by https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation.

    Attributes
    ----------
    output_layer_name: str
        Name of the output layer relevant to the parser.
    score_threshold : float
        Confidence score threshold for detected keypoints.

    Output Message/s
    ----------------
    **Type**: Keypoints

    **Description**: Keypoints message containing detected body keypoints.
    """

    def __init__(
        self, output_layer_name: str = "", score_threshold: float = 0.5
    ) -> None:
        """Initializes the parser node.

        @param output_layer_name: Name of the output layer relevant to the parser.
        @type output_layer_name: str
        @param score_threshold: Confidence score threshold for detected keypoints.
        @type score_threshold: float
        """
        super().__init__(output_layer_name, score_threshold=score_threshold)

    def setOutputLayerName(self, output_layer_name: str) -> None:
        """Sets the name of the output layer.

        @param output_layer_name: The name of the output layer.
        @type output_layer_name: str
        """
        if not isinstance(output_layer_name, str):
            raise ValueError("Output layer name must be a string.")
        self.output_layer_name = output_layer_name

    def build(
        self,
        head_config: Dict[str, Any],
    ) -> "HRNetParser":
        """Configures the parser.

        @param head_config: The head configuration for the parser.
        @type head_config: Dict[str, Any]
        @return: The parser object with the head configuration set.
        @rtype: HRNetParser
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

            heatmaps = output.getTensor(
                self.output_layer_name,
                dai.TensorInfo.StorageOrder.NCHW,  # this signals DAI to return output as NCHW
                dequantize=True,
            )

            if heatmaps.shape[0] == 1:
                heatmaps = heatmaps[0]  # remove batch dimension

            if len(heatmaps.shape) != 3:
                raise ValueError(
                    f"Expected 3D output tensor, got {len(heatmaps.shape)}D."
                )

            self.n_keypoints, map_h, map_w = heatmaps.shape

            scores = np.array([np.max(heatmap) for heatmap in heatmaps])
            scores = np.clip(scores, 0, 1)  # TODO: check why scores are sometimes >1

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
            keypoints_message.transformation = output.getTransformation()

            self.out.send(keypoints_message)

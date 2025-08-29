from typing import Any, Dict, List, Optional, Tuple

import depthai as dai
import numpy as np

from depthai_nodes.message.creators import create_keypoints_message
from depthai_nodes.node.parsers.keypoints import KeypointParser
from depthai_nodes.node.parsers.utils.superanimal import get_pose_prediction


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
    label_names : List[str]
        Label names for the keypoints.
    edges : List[Tuple[int, int]]
        Keypoint connection pairs for visualizing the skeleton. Example: [(0,1), (1,2), (2,3), (3,0)] shows that keypoint 0 is connected to keypoint 1, keypoint 1 is connected to keypoint 2, etc.

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
        label_names: Optional[List[str]] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
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
        @param label_names: Label names for the keypoints.
        @type label_names: Optional[List[str]]
        @param edges: Keypoint connection pairs for visualizing the skeleton. Example:
            [(0,1), (1,2), (2,3), (3,0)] shows that keypoint 0 is connected to keypoint
            1, keypoint 1 is connected to keypoint 2, etc.
        @type edges: Optional[List[Tuple[int, int]]]
        """
        super().__init__(
            output_layer_name,
            scale_factor=scale_factor,
            n_keypoints=n_keypoints,
            score_threshold=score_threshold,
            label_names=label_names,
            edges=edges,
        )
        self._logger.debug(
            f"SuperAnimalParser initialized with output_layer_name='{output_layer_name}', scale_factor={scale_factor}, n_keypoints={n_keypoints}, score_threshold={score_threshold}, label_names={label_names}, edges={edges}"
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

        self._logger.debug("SuperAnimalParser built")

        return self

    def run(self):
        self._logger.debug("SuperAnimalParser run started")
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            layers = output.getAllLayerNames()
            self._logger.debug(f"Processing input with layers: {layers}")
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

            msg = create_keypoints_message(
                keypoints,
                scores,
                self.score_threshold,
                label_names=self.label_names,
                edges=self.edges,
            )
            msg.setTimestamp(output.getTimestamp())
            msg.setSequenceNum(output.getSequenceNum())
            msg.setTimestampDevice(output.getTimestampDevice())
            transformation = output.getTransformation()
            if transformation is not None:
                msg.setTransformation(transformation)

            self._logger.debug(
                f"Created keypoints message with {len(keypoints)} points"
            )

            self.out.send(msg)

            self._logger.debug("Keypoints message sent successfully")

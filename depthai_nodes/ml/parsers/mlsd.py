from typing import Any, Dict

import depthai as dai
import numpy as np

from ..messages.creators import create_line_detection_message
from .base_parser import BaseParser
from .utils.mlsd import decode_scores_and_points, get_lines


class MLSDParser(BaseParser):
    """Parser class for parsing the output of the M-LSD line detection model. The parser
    is specifically designed to parse the output of the M-LSD model. As the result, the
    node sends out the detected lines in the form of a message.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    output_layer_tpmap : str
        Name of the output layer containing the tpMap tensor.
    output_layer_heat : str
        Name of the output layer containing the heat tensor.
    topk_n : int
        Number of top candidates to keep.
    score_thr : float
        Confidence score threshold for detected lines.
    dist_thr : float
        Distance threshold for merging lines.

    Output Message/s
    ----------------
    **Type**: LineDetections

    **Description**: LineDetections message containing detected lines and confidence scores.
    """

    def __init__(
        self,
        output_layer_tpmap="",
        output_layer_heat="",
        topk_n=200,
        score_thr=0.10,
        dist_thr=20.0,
    ):
        """Initializes the MLSDParser node.

        @param topk_n: Number of top candidates to keep.
        @type topk_n: int
        @param score_thr: Confidence score threshold for detected lines.
        @type score_thr: float
        @param dist_thr: Distance threshold for merging lines.
        @type dist_thr: float
        """
        super().__init__()
        self.output_layer_tpmap = output_layer_tpmap
        self.output_layer_heat = output_layer_heat

        self.topk_n = topk_n
        self.score_thr = score_thr
        self.dist_thr = dist_thr

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
        MLSDParser
            Returns the parser object with the head configuration set.
        """

        output_layers = head_config["outputs"]
        if len(output_layers) != 2:
            raise ValueError(
                f"Only two output layers are supported for MLSDParser, got {len(output_layers)} layers."
            )
        for layer in output_layers:
            if "tpMap" in layer:
                self.output_layer_tpmap = layer
            elif "heat" in layer:
                self.output_layer_heat = layer
        self.topk_n = head_config["topk_n"]
        self.score_thr = head_config["score_thr"]
        self.dist_thr = head_config["dist_thr"]

        return self

    def setOutputLayerTPMap(self, output_layer_tpmap):
        """Sets the name of the output layer containing the tpMap tensor.

        @param output_layer_tpmap: Name of the output layer containing the tpMap tensor.
        @type output_layer_tpmap: str
        """
        self.output_layer_tpmap = output_layer_tpmap

    def setOutputLayerHeat(self, output_layer_heat):
        """Sets the name of the output layer containing the heat tensor.

        @param output_layer_heat: Name of the output layer containing the heat tensor.
        @type output_layer_heat: str
        """
        self.output_layer_heat = output_layer_heat

    def setTopK(self, topk_n):
        """Sets the number of top candidates to keep.

        @param topk_n: Number of top candidates to keep.
        @type topk_n: int
        """
        self.topk_n = topk_n

    def setScoreThreshold(self, score_thr):
        """Sets the confidence score threshold for detected lines.

        @param score_thr: Confidence score threshold for detected lines.
        @type score_thr: float
        """
        self.score_thr = score_thr

    def setDistanceThreshold(self, dist_thr):
        """Sets the distance threshold for merging lines.

        @param dist_thr: Distance threshold for merging lines.
        @type dist_thr: float
        """
        self.dist_thr = dist_thr

    def run(self):
        if self.output_layer_tpmap == "":
            raise ValueError(
                "Output layer containing the tpMap tensor is not set. Please use setOutputLayerTPMap method or correct NN archive."
            )
        if self.output_layer_heat == "":
            raise ValueError(
                "Output layer containing the heat tensor is not set. Please use setOutputLayerHeat method or correct NN archive."
            )

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            tpMap = output.getTensor(self.output_layer_tpmap, dequantize=True).astype(
                np.float32
            )
            heat_np = output.getTensor(self.output_layer_heat, dequantize=True).astype(
                np.float32
            )

            if len(tpMap.shape) != 4:
                raise ValueError("Invalid shape of the tpMap tensor. Should be 4D.")
            if tpMap.shape[3] == 9:
                # We have NWHC format, transform to NCHW
                tpMap = np.transpose(tpMap, (0, 3, 1, 2))

            pts, pts_score, vmap = decode_scores_and_points(tpMap, heat_np, self.topk_n)
            lines, scores = get_lines(
                pts, pts_score, vmap, self.score_thr, self.dist_thr
            )

            message = create_line_detection_message(lines, np.array(scores))
            message.setTimestamp(output.getTimestamp())
            self.out.send(message)

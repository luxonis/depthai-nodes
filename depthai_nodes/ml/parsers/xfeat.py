from typing import Tuple

import depthai as dai
import numpy as np

from ..messages.creators import create_tracked_features_message
from .utils.xfeat import detect_and_compute, match


class XFeatParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the XFeat model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    original_size : Tuple[float, float]
        Original image size.
    input_size : Tuple[float, float]
        Input image size.
    max_keypoints : int
        Maximum number of keypoints to keep.
    previous_results : np.ndarray
        Previous results from the model. Previous results are used to match keypoints between two frames.

    Output Message/s
    ----------------
    **Type**: dai.TrackedFeatures

    **Description**: TrackedFeatures message containing matched keypoints with the same ID.

    Error Handling
    --------------
    **ValueError**: If the original image size is not specified.
    """

    def __init__(
        self,
        original_size: Tuple[float, float] = None,
        input_size: Tuple[float, float] = (640, 352),
        max_keypoints: int = 4096,
    ):
        """Initializes the XFeatParser node.

        @param original_size: Original image size.
        @type original_size: Tuple[float, float]
        @param input_size: Input image size.
        @type input_size: Tuple[float, float]
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()
        self.original_size = original_size
        self.input_size = input_size
        self.max_keypoints = max_keypoints
        self.previous_results = None

    def setOriginalSize(self, original_size):
        """Sets the original image size.

        @param original_size: Original image size.
        @type original_size: Tuple[float, float]
        """
        self.original_size = original_size

    def setInputSize(self, input_size):
        """Sets the input image size.

        @param input_size: Input image size.
        @type input_size: Tuple[float, float]
        """
        self.input_size = input_size

    def setMaxKeypoints(self, max_keypoints):
        """Sets the maximum number of keypoints to keep.

        @param max_keypoints: Maximum number of keypoints.
        @type max_keypoints: int
        """
        self.max_keypoints = max_keypoints

    def run(self):
        if self.original_size is None:
            raise ValueError("Original image size must be specified!")

        resize_rate_w = self.original_size[0] / self.input_size[0]
        resize_rate_h = self.original_size[1] / self.input_size[1]

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            feats = output.getTensor("feats", dequantize=True).astype(np.float32)
            keypoints = output.getTensor("keypoints", dequantize=True).astype(
                np.float32
            )

            if len(feats.shape) == 3:
                feats = feats.reshape((1,) + feats.shape).transpose(0, 3, 1, 2)
            if len(keypoints.shape) == 3:
                keypoints = keypoints.reshape((1,) + keypoints.shape).transpose(
                    0, 3, 1, 2
                )

            result = detect_and_compute(
                feats,
                keypoints,
                resize_rate_w,
                resize_rate_h,
                self.input_size,
                self.max_keypoints,
            )

            if result is not None:
                result = result[0]
            else:
                matched_points = dai.TrackedFeatures()
                matched_points.setTimestamp(output.getTimestamp())
                self.out.send(matched_points)
                continue

            if self.previous_results is not None:
                mkpts0, mkpts1 = match(self.previous_results, result)
                matched_points = create_tracked_features_message(mkpts0, mkpts1)
                matched_points.setTimestamp(output.getTimestamp())
                self.out.send(matched_points)

            # save the result from first frame
            self.previous_results = result

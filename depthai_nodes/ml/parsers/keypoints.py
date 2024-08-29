import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message


class KeypointParser(dai.node.ThreadedHostNode):
    """Parser class for 2D or 3D keypoints models. It expects one ouput layer containing
    keypoints. The number of keypoints must be specified. Moreover, the keypoints are
    normalized by a scale factor if provided.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.
    scale_factor : float
        Scale factor to divide the keypoints by.
    n_keypoints : int
        Number of keypoints the model detects.

    Output Message/s
    ----------------
    **Type**: Keypoints

    **Description**: Keypoints message containing 2D or 3D keypoints.

    Error Handling
    --------------
    **ValueError**: If the number of keypoints is not specified.

    **ValueError**: If the number of coordinates per keypoint is not 2 or 3.

    **ValueError**: If the number of output layers is not 1.
    """

    def __init__(
        self,
        scale_factor=1,
        n_keypoints=None,
    ):
        """Initializes KeypointParser node.

        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

        self.scale_factor = scale_factor
        self.n_keypoints = n_keypoints

    def setScaleFactor(self, scale_factor):
        """Sets the scale factor to divide the keypoints by.

        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        """
        self.scale_factor = scale_factor

    def setNumKeypoints(self, n_keypoints):
        """Sets the number of keypoints.

        @param n_keypoints: Number of keypoints.
        @type n_keypoints: int
        """
        self.n_keypoints = n_keypoints

    def run(self):
        if self.n_keypoints is None:
            raise ValueError("Number of keypoints must be specified!")

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()

            if len(output_layer_names) != 1:
                raise ValueError(
                    f"Expected 1 output layer, got {len(output_layer_names)}."
                )

            keypoints = output.getTensor(output_layer_names[0], dequantize=True).astype(
                np.float32
            )
            num_coords = int(np.prod(keypoints.shape) / self.n_keypoints)

            if num_coords not in [2, 3]:
                raise ValueError(
                    f"Expected 2 or 3 coordinates per keypoint, got {num_coords}."
                )

            keypoints = keypoints.reshape(self.n_keypoints, num_coords)

            keypoints /= self.scale_factor

            msg = create_keypoints_message(keypoints)
            msg.setTimestamp(output.getTimestamp())

            self.out.send(msg)

import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message


class KeypointParser(dai.node.ThreadedHostNode):
    """KeypointParser class for 2D or 3D keypoints models."""

    def __init__(
        self,
        scale_factor=1,
        num_keypoints=None,
    ):
        """Initializes KeypointParser node with input, output, scale factor, and number
        of keypoints.

        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        @param num_keypoints: Number of keypoints.
        @type num_keypoints: int
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.scale_factor = scale_factor
        self.num_keypoints = num_keypoints

    def setScaleFactor(self, scale_factor):
        """Sets the scale factor to divide the keypoints by.

        @param scale_factor: Scale factor to divide the keypoints by.
        @type scale_factor: float
        """
        self.scale_factor = scale_factor

    def setNumKeypoints(self, num_keypoints):
        """Sets the number of keypoints.

        @param num_keypoints: Number of keypoints.
        @type num_keypoints: int
        """
        self.num_keypoints = num_keypoints

    def run(self):
        """Function executed in a separate thread that processes the input data and
        sends it out in form of messages.

        @raises ValueError: If the number of keypoints is not specified.
        @raises ValueError: If the number of coordinates per keypoint is not 2 or 3.
        @raises ValueError: If the number of output layers is not 1.
        @return: Keypoints message containing 2D or 3D keypoints.
        """

        if self.num_keypoints is None:
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

            keypoints = output.getTensor(output_layer_names[0])
            num_coords = int(np.prod(keypoints.shape) / self.num_keypoints)

            if num_coords not in [2, 3]:
                raise ValueError(
                    f"Expected 2 or 3 coordinates per keypoint, got {num_coords}."
                )

            keypoints = keypoints.reshape(self.num_keypoints, num_coords)

            keypoints /= self.scale_factor

            msg = create_keypoints_message(keypoints)

            self.out.send(msg)

import depthai as dai
import numpy as np

from ..messages.creators import create_keypoints_message

class KeypointParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        scale_factor=192,
        num_keypoints=468,
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.scale_factor = scale_factor
        self.num_keypoints = num_keypoints

    def setScaleFactor(self, scale_factor):
        self.scale_factor = scale_factor

    def setNumKeypoints(self, num_keypoints):
        self.num_keypoints = num_keypoints

    def run(self):
        """
        Postprocessing logic for Keypoint model.

        Returns:
            dai.Keypoints: num_keypoints keypoints (2D or 3D).
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()
            
            if len(output_layer_names) != 1:
                raise ValueError(f"Expected 1 output layer, got {len(output_layer_names)}.")
            
            keypoints = output.getTensor(output_layer_names[0])
            num_coords = int(np.prod(keypoints.shape) / self.num_keypoints)
            
            if num_coords not in [2, 3]:
                raise ValueError(f"Expected 2 or 3 coordinates per keypoint, got {num_coords}.")
            
            keypoints = keypoints.reshape(self.num_keypoints, num_coords)

            keypoints /= self.scale_factor

            msg = create_keypoints_message(keypoints)

            self.out.send(msg)

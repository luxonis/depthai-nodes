import depthai as dai

from ..messages.creators import create_depth_message


class MonocularDepthParser(dai.node.ThreadedHostNode):
    def __init__(self, depth_type="relative"):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.depth_type = depth_type

    def setRelativeDepthType(self):
        self.depth_type = "relative"

    def setMetricDepthType(self):
        self.depth_type = "metric"

    def run(self):
        """
        Postprocessing logic for a model with monocular depth output (e.g.Depth Anything model).

        Returns:
            dai.ImgFrame: uint16, HW depth map.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()
            if len(output_layer_names) != 1:
                raise ValueError(
                    f"Expected 1 output layer, got {len(output_layer_names)}."
                )
            output = output.getTensor(output_layer_names[0])

            depth_map = output[0]

            depth_message = create_depth_message(
                depth_map=depth_map,
                depth_type=self.depth_type,
            )
            self.out.send(depth_message)

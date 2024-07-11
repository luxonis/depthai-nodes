import depthai as dai

from ..messages.creators import create_thermal_message


class ThermalImageParser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """Postprocessing logic for a model with thermal image output (e.g. UGSR-FA).

        Returns:
            dai.ImgFrame: uint16, HW thermal image.
        """

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
            output = output.getTensor(output_layer_names[0])
           
            thermal_map = output[0]

            depth_message = create_thermal_message(
                thermal_map=thermal_map
            )
            self.out.send(depth_message)

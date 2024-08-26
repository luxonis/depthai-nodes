import depthai as dai

from ..messages.creators import create_thermal_message


class ThermalImageParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of models with thermal image output (e.g.
    UGSR-FA).

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.

    Output Message/s
    ----------------
    **Type**: dai.ImgFrame

    **Description**: Thermal message containing the thermal image.
    """

    def __init__(self):
        """Initializes the ThermalImageParser node."""
        dai.node.ThreadedHostNode.__init__(self)
        self.input = self.createInput()
        self.out = self.createOutput()

    def run(self):
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

            thermal_message = create_thermal_message(thermal_map=thermal_map)
            thermal_message.setTimestamp(output.getTimestamp())
            self.out.send(thermal_message)

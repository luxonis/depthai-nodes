import depthai as dai

from ..messages.creators import create_density_message


class DMCountParser(dai.node.ThreadedHostNode):
    """Parser class for parsing the output of the DM-Count crowd density estimation
    model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.

    Output Message/s
    ----------------
    **Type**: Map2D

    **Description**: Density message containing the density map. The density map is represented with Map2D object.
    """

    def __init__(
        self,
    ):
        """Initializes the DMCountParser node."""
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

            density_map = output.getTensor(output_layer_names[0], dequantize=True)[0, 0]

            density_message = create_density_message(density_map)

            self.out.send(density_message)

import depthai as dai

from ..messages.creators import create_vehicle_attributes_message


class VehicleAttributesParser(dai.node.ThreadedHostNode):
    """Postprocessing logic for Vehicle Attributes model.

    Attributes
    ----------
    input : Node.Input
        Node's input. It is a linking point to which the Neural Network's output is linked. It accepts the output of the Neural Network node.
    out : Node.Output
        Parser sends the processed network results to this output in a form of DepthAI message. It is a linking point from which the processed network results are retrieved.

    Output Message/s
    ----------------
    **Type**: VehicleAttributes

    **Description**: Message containing two tuples like (class_name, probability). First is `vehicle_type` and second is `vehicle_color`.
    """

    def __init__(self):
        """Initializes the MultipleClassificationParser node."""
        dai.node.ThreadedHostNode.__init__(self)
        self.out = self.createOutput()
        self.input = self.createInput()

    def run(self):
        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break

            layer_names = output.getAllLayerNames()

            vehicle_types = output.getTensor(layer_names[0], dequantize=True)
            vehicle_colors = output.getTensor(layer_names[1], dequantize=True)

            vehicle_types = vehicle_types.flatten().tolist()
            vehicle_colors = vehicle_colors.flatten().tolist()

            vehicle_attributes_message = create_vehicle_attributes_message(
                vehicle_types, vehicle_colors
            )
            vehicle_attributes_message.setTimestamp(output.getTimestamp())

            self.out.send(vehicle_attributes_message)

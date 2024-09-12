import depthai as dai

from ..messages.creators import create_vehicle_attributes_message


class VehicleAttributesParser(dai.node.ThreadedHostNode):
    """Postprocessing logic for Vehicle Attributes model."""

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

            output_layer_names = output.getAllLayerNames()

            vehicle_types = (
                output.getTensor(output_layer_names[0], dequantize=True)
                .flatten()
                .tolist()
            )
            vehicle_colors = (
                output.getTensor(output_layer_names[1], dequantize=True)
                .flatten()
                .tolist()
            )

            vehicle_attributes_message = create_vehicle_attributes_message(
                vehicle_types, vehicle_colors
            )
            vehicle_attributes_message.setTimestamp(output.getTimestamp())

            self.out.send(vehicle_attributes_message)

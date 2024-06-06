import depthai as dai

from .utils.message_creation import create_monocular_depth_msg


class DepthAnythingParser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """
        Postprocessing logic for Depth Anything model.

        Returns:
            dai.ImgFrame: uint16, HW depth map.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            output = output.getTensor(
                "depth"
            )  # numpy.ndarray of shape (1, 1, 518, 518)

            depth_message = create_monocular_depth_msg(
                depth_map=output[0, 0],
                depth_type="relative",
            )
            self.out.send(depth_message)

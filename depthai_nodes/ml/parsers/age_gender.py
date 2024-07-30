import depthai as dai

from ..messages.creators import create_age_gender_message


class AgeGenderParser(dai.node.ThreadedHostNode):
    def __init__(
        self
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """Postprocessing logic for age_gender model.

        Returns:
            Detected person age and gender probability.
        """

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            age = output.getTensor("age_conv3", dequantize=True)
            prob = output.getTensor("prob", dequantize=True)

            age_gender_message = create_age_gender_message(
                age=age.item(), 
                gender_prob=prob.flatten().tolist()
            )

            self.out.send(age_gender_message)

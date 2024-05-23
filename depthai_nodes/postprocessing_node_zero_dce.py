import depthai as dai
import numpy as np
import cv2


class ZeroDCEParser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """ Postprocessing logic for Zero-DCE model. Output is a BGR. HWC, uint8 image."""
        
        while self.isRunning():

            try:
                output : dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break # Pipeline was stopped

            output = output.getTensor("87") # numpy.ndarray of shape (1, 3, 400, 600)
            output = output[0]
            #un-normalize
            output = output * 255
            # convert back to uint8
            output = output.astype(np.uint8)
            # CHW to HWC
            output = np.transpose(output, (1,2,0))
            # RGB to BGR
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            imgFrame = dai.ImgFrame()
            imgFrame.setFrame(output)
            imgFrame.setWidth(output.shape[1])
            imgFrame.setHeight(output.shape[0])
            imgFrame.setType(dai.ImgFrame.Type.BGR888i)

            self.out.send(imgFrame)
        
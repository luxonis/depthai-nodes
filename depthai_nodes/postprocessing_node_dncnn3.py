import depthai as dai
import numpy as np


class DnCNN3Parser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """ 
        Postprocessing logic for DnCNN3 model. 
        
        Returns:
            dai.ImgFrame: uint8, GRAYSCALE image representing the denoised image.
        """
        
        while self.isRunning():

            try:
                output : dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break # Pipeline was stopped

            output = output.getTensor("80") # numpy.ndarray of shape (1, 1, 321, 481)
            output = output[0][0]

            #un-normalize
            output = output*255
            # convert back to uint8
            output = output.astype(np.uint8)

            imgFrame = dai.ImgFrame()
            imgFrame.setFrame(output)
            imgFrame.setWidth(output.shape[1])
            imgFrame.setHeight(output.shape[0])
            imgFrame.setType(dai.ImgFrame.Type.GRAY8)

            self.out.send(imgFrame)
        
import depthai as dai
import numpy as np
import cv2


class DepthAnythingParser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """ 
        Postprocessing logic for Depth Anything model. 
        
        Returns:
            dai.ImgFrame: uint8, HWC image representing the depth colormap.
        """
        
        while self.isRunning():

            try:
                output : dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break # Pipeline was stopped

            output = output.getTensor("depth") # numpy.ndarray of shape (1, 1, 518, 518)
            output = output[0][0]

            # un-normalize
            output = (output - output.min()) / (output.max() - output.min()) * 255.0
            # convert to uint8
            output = output.astype(np.uint8)
            # apply colormapping
            output = cv2.applyColorMap(output, cv2.COLORMAP_INFERNO)
            # TODO: exclude the following step from postprocessing?
            # i.e. do we want as output colormap representation of depth?

            imgFrame = dai.ImgFrame()
            imgFrame.setFrame(output)
            imgFrame.setWidth(output.shape[1])
            imgFrame.setHeight(output.shape[0])
            imgFrame.setType(dai.ImgFrame.Type.BGR888i)

            self.out.send(imgFrame)
        
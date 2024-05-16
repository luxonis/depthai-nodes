import depthai as dai


class CustomParser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """ Define postprocessing logic here. """
        
        while self.isRunning():
            try:
                nnDataIn : dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break # Pipeline was stopped, no more data
            print(f"Layer names = {nnDataIn.getAllLayerNames()}")
            # TODO implement the actual parsing
            detectionMessage = dai.ImgDetections()
            exampleDetection = dai.ImgDetection()
            exampleDetection.label = 0
            exampleDetection.confidence = 0.5
            exampleDetection.xmin = 0.1
            exampleDetection.ymin = 0.1
            exampleDetection.xmax = 0.9
            exampleDetection.ymax = 0.9
            detectionMessage.detections = [exampleDetection]
            self.out.send(detectionMessage)
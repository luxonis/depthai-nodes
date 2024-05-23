import depthai as dai


class CustomParser(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """ Define postprocessing logic here. It runs on host computer, not device! 
        So feel free to use any python libraries you want (but preferably stick to numpy and cv2) """
        
        while self.isRunning():
            try:
                nnDataIn : dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break # Pipeline was stopped, no more data
            print(f"Layer names = {nnDataIn.getAllLayerNames()}")
            # TODO implement the postprocessing logic
            # TODO: send out the processed data
            #self.out.send(postprocessed_output)
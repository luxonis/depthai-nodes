import cv2
import depthai as dai
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from preprocessing import *
from ml.postprocessing import *
from saving import *

preprocess_functions = {
    'zero_dce': preprocess_zero_dce,
}

parsers = {
    'zero_dce': ImageOutputParser,
}

frame_types = {
    'zero_dce': dai.ImgFrame.Type.BGR888p,
}

save_functions = {
    'zero_dce': save_zero_dce_output,
}

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--model_name", help="model name", required=True, type=str)
parser.add_argument("-model", "--model_path", help="blob model path", required=True, type=str)
parser.add_argument("-iw", "--input_width", help="blob model input width", required=True, type=int)
parser.add_argument("-ih", "--input_height", help="blob model input height", required=True, type=int)
parser.add_argument('-img', '--image_path', help="path to input image", required=True, type=str)
args = parser.parse_args()

model_name = args.model_name
model_path = args.model_path
input_width = args.input_width
input_height = args.input_height
image_path = args.image_path


# --------------- Define Input Node ---------------
class HostImage(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.out = dai.Node.Output(self)
    
    def run(self):
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (input_width, input_height))
        frame = preprocess_functions[model_name](frame)
        
         # Create ImgFrame message
        imgFrame = dai.ImgFrame()
        _, frame_height, frame_width = frame.shape
        imgFrame.setFrame(frame)
        imgFrame.setWidth(frame_width)
        imgFrame.setHeight(frame_height)
        imgFrame.setType(frame_types[model_name])
        
        self.out.send(imgFrame)


# --------------- Pipeline ---------------
with dai.Pipeline() as pipeline:
    # Creation
    inFrame = pipeline.create(HostImage)
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(model_path)
    detection_postprocessor = pipeline.create(parsers[model_name])
    
    # Linking
    inFrame.out.link(detection_nn.input)
    detection_nn.out.link(detection_postprocessor.input)
    
    # Queueing
    frameQueue = detection_nn.passthrough.createQueue()
    postprocessorQueue = detection_postprocessor.out.createQueue()

    pipeline.start()

    while pipeline.isRunning():
        save_functions[model_name](postprocessorQueue.get())
        pipeline.stop()
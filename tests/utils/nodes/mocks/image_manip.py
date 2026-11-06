from .input import InputMock
from .output import OutputMock


class ImageManipConfigMock:
    def __init__(self):
        self.outputSize = None
        self.frameType = None

    def setOutputSize(self, width, height, mode=None):
        self.outputSize = (width, height, mode)

    def setFrameType(self, frame_type):
        self.frameType = frame_type


class ImageManipMock:
    def __init__(self):
        self.initialConfig = ImageManipConfigMock()
        self.inputImage = InputMock()
        self._out = OutputMock()

    @property
    def out(self):
        return self._out

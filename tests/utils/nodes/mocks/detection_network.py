from .input import InputMock
from .output import OutputMock
from .pipeline import DetectionParserMock


class DetectionNetworkMock:
    """Stand-in for ``dai.node.DetectionNetwork``: a network and its parser."""

    def __init__(self):
        self._input = InputMock()
        self._out = OutputMock()
        self._nn_archive = None
        self.detectionParser = DetectionParserMock()

    @property
    def input(self):
        return self._input

    @property
    def out(self):
        return self._out

    def build(self, input, nnArchive, confidenceThreshold: float = 0.5):
        self._nn_archive = nnArchive
        input.link(self.input)
        self.detectionParser.setNNArchive(nnArchive)
        self.detectionParser.setConfidenceThreshold(confidenceThreshold)
        return self

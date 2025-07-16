from typing import Union

import depthai as dai

from .input import InputMock
from .output import OutputMock


class NeuralNetworkMock:
    def __init__(self):
        self._input = InputMock()
        self._out = OutputMock()
        self._passthrough = OutputMock()
        self._nn_archive = None

    def build(
        self,
        input: InputMock,
        model: Union[dai.NNModelDescription, dai.NNArchive],
        fps: float,
    ):
        self._nn_archive = model
        self._input = input
        return self

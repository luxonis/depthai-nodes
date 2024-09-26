from abc import ABCMeta, abstractmethod
from typing import Dict, Union

import depthai as dai


class BaseMeta(ABCMeta, type(dai.node.ThreadedHostNode)):
    pass


class BaseParser(dai.node.ThreadedHostNode, metaclass=BaseMeta):
    """Base parser class for neural network output parsers."""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def input(self) -> dai.Node.Input:
        pass

    @property
    @abstractmethod
    def output(self) -> dai.Node.Output:
        pass

    @abstractmethod
    def build(self, head: Union[dai.NNArchive, Dict]):
        pass

    @abstractmethod
    def run(self):
        pass

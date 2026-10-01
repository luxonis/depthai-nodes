"""Host nodes, neural network parsers, and message helpers for DepthAI v3.

Use ``depthai_nodes.node`` to build pipeline nodes, ``depthai_nodes.message`` for custom
containers and native message creators, and ``depthai_nodes.runtime`` for runtime
integrations. Parser helpers select native DepthAI parsers by default; use
``HostParsingNeuralNetwork`` for the Python implementations.
"""

from .constants import *
from .logging import setup_logging
from .message import *

__version__ = "0.6.2"


setup_logging()

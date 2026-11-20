"""Host nodes, neural network parsers, and message helpers for DepthAI v3.

Use ``depthai_nodes.node`` to build pipeline nodes, ``depthai_nodes.message`` for custom
containers and native message creators, and ``depthai_nodes.runtime`` for runtime
integrations. Parser helpers select native DepthAI parsers by default; use
``HostParsingNeuralNetwork`` for the Python implementations.
"""

from .constants import *
from .logging import setup_logging
from .message import *
from .support_plane_store import (
    ActiveSupportPlane as ActiveSupportPlane,
)
from .support_plane_store import (
    SupportPlaneStore as SupportPlaneStore,
)
from .support_plane_store import (
    resolve_support_plane_path as resolve_support_plane_path,
)

__version__ = "0.8.0"


setup_logging()

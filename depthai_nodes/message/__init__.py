"""Custom DepthAI message containers.

``Collection`` holds items of one runtime type. ``GatheredData`` associates a collection
with a reference message and copies its timestamps and sequence number. ``SnapData``
carries an image file group and metadata for snap uploads.

Functions in ``depthai_nodes.message.creators`` produce native DepthAI messages for
parser outputs. Functions in ``depthai_nodes.message.utils`` copy messages and compute
detection geometry.
"""

from .collection import Collection
from .gathered_data import GatheredData
from .snap_data import SnapData

__all__ = [
    "GatheredData",
    "SnapData",
    "Collection",
]

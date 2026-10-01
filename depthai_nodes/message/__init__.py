"""Native parser messages and custom messages for DepthAI v3.

Parser creators return native DepthAI messages. The package requires DepthAI
3.9 or newer, which provides the beta message types used by the host parsers.
Use `depthai_nodes.message.creators` to construct messages from arrays and
`depthai_nodes.message.utils` for copying and detection geometry.

.. contents:: Contents
   :depth: 2

Native parser messages
======================

The creator functions document the required shapes, coordinate conventions,
and value ranges for each payload:

* ``dai.ImgDetections`` contains detections, including optional keypoints and
  instance masks; ``dai.SpatialImgDetections`` adds spatial coordinates.
* ``dai.SegmentationMask`` contains a segmentation mask.
* ``dai.beta.Classifications`` contains class names and scores, including
  sequence classification results.
* ``dai.beta.Clusters`` and ``dai.beta.Cluster`` represent grouped 2D points.
* ``dai.beta.Keypoints`` contains keypoints and optional skeleton edges.
* ``dai.beta.Lines`` and ``dai.beta.Line`` represent line detections.
* ``dai.beta.Map2D`` stores a numeric map, such as a depth estimate.
* ``dai.beta.Predictions`` and ``dai.beta.Prediction`` store regression results.
* ``dai.TrackedFeatures`` represents feature points and matches.
* ``dai.ImgFrame`` and ``dai.NNData`` carry images and embedding tensors.

Custom messages
===============

`Collection` stores items of one runtime type. Add items with ``append()``
or ``extend()``. The first item establishes the accepted type; an initially
empty collection infers its type when populated. Assigning ``items`` validates
the replacement list, but directly mutating the returned list bypasses that
validation. ``copy()`` copies each item through ``copy_message`` and preserves
the message timestamps and sequence number.

`GatheredData` extends `Collection` with a reference message. Assigning
``reference_data`` copies its timestamps and sequence number to the message.
`depthai_nodes.node.GatherData` produces these messages by matching messages
around a reference timestamp.

`SnapData` stores a snap name, a ``dai.FileGroup`` containing the image and
associated files, optional tags, and string metadata. Pass these messages to
`depthai_nodes.node.SnapsUploader` to upload them through the Hub Events API.

Example:
    Group detections with the frame they refer to::

        import depthai as dai
        from depthai_nodes.message import GatheredData

        frame = dai.ImgFrame()
        detections = dai.ImgDetections()
        batch = GatheredData(reference_data=frame, items=[detections])
        assert batch.reference_data is frame

Note:
    Creator functions populate payloads. Callers that emit messages from a
    pipeline must also set the appropriate timestamps, sequence number, and
    image transformation. Native message copies may share pixel storage; see
    `depthai_nodes.message.utils.copy_message` before modifying copied buffers.
"""

from .collection import Collection
from .gathered_data import GatheredData
from .snap_data import SnapData

__all__ = [
    "GatheredData",
    "SnapData",
    "Collection",
]

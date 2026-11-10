"""Native parser messages and custom messages for DepthAI v3.

Parser creators return native DepthAI messages. The package requires DepthAI
3.11 or newer. Use `depthai_nodes.message.creators` to construct messages from
arrays and `depthai_nodes.message.utils` for copying and detection geometry.

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

Support plane
-------------

`SupportPlane` is the support plane equation ``normal · point + offset_mm = 0``
in millimeters, normalized on construction. ``toJson()``, ``fromJson()``,
``toJsonFile()``, and ``fromJsonFile()`` save and load exactly its ``normal`` and
``offset_mm``.

`depthai_nodes.node.SupportPlaneCalibrator` exchanges these messages:

* `SupportPlaneCalibrationControl` starts, stops, or resets calibration, or
  asks for a plane verification of a given support plane.
* `SupportPlaneCalibrationResult` ends a calibration attempt with a
  `CalibrationOutcome` and, only when it succeeded, its support plane
  candidates, best first.
* `SupportPlaneCandidate` holds a ``plane`` with its ``inlier_count`` and
  ``inlier_ratio``.
* `SupportPlaneVerificationResult` reports the result of a plane verification:
  the plane's angle to gravity, whether it was accepted, and its on-plane
  ratio, the share of the checked point cloud's sampled points that lie on the
  plane (``None`` when not measured). The on-plane ratio is not a candidate's
  inlier ratio.

The fields of these messages are read-only. `SupportPlane` and
`SupportPlaneCandidate` are frozen dataclasses that take keyword arguments and
raise ``TypeError`` for a wrong type or ``ValueError`` for an invalid value.

Box measurement
---------------

`BoxDetections` is a `Collection` of `BoxDetection` items, one box detection
per tracked box: its track ID, normalized bounding box, confidence, and tracking
status. A shared ``uint8`` segmentation mask covers the whole frame; a pixel
holds the index of the item it belongs to, and 255 marks background.
``getMask()`` returns one item's boolean mask.

`BoxMeasurements` is a `Collection` of `BoxMeasurementResult` items. Each result
repeats the box detection fields and adds a `MeasurementStatus` and, when the
box has any, its `BoxDimensions`: length, width, and height in millimeters. A
measured result always has dimensions. Both messages draw labelled box outlines
through ``getVisualizationMessage()``. `BoxDetection`, `BoxMeasurementResult`,
and `BoxDimensions` are frozen dataclasses that take keyword arguments and
raise the same errors as the support plane value types.

Boxes are measured against a support plane; see "Support plane" above.
"""

from .box_detections import BoxDetection, BoxDetections
from .box_measurements import (
    BoxDimensions,
    BoxMeasurementResult,
    BoxMeasurements,
    MeasurementStatus,
)
from .collection import Collection
from .gathered_data import GatheredData
from .snap_data import SnapData
from .support_plane import SupportPlane, SupportPlaneCandidate
from .support_plane_calibration import (
    CalibrationOutcome,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
    SupportPlaneVerificationResult,
)

__all__ = [
    "GatheredData",
    "SnapData",
    "Collection",
    "BoxDetections",
    "BoxDetection",
    "BoxMeasurements",
    "BoxMeasurementResult",
    "BoxDimensions",
    "MeasurementStatus",
    "SupportPlane",
    "SupportPlaneCandidate",
    "SupportPlaneCalibrationControl",
    "SupportPlaneCalibrationResult",
    "SupportPlaneVerificationResult",
    "CalibrationOutcome",
]

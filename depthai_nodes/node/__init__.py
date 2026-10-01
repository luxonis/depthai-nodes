"""Pipeline nodes for inference, postprocessing, and host-side message handling.

Create nodes with ``pipeline.create(...)`` and connect their input and output
ports. The classes in this package integrate native DepthAI inference and
parsers with Python postprocessing, image utilities, and custom message handling.

.. contents:: Contents
   :depth: 2

Inference and parser selection
==============================

`ParsingNeuralNetwork` builds a neural network and the native DepthAI parsers
specified by an NN Archive. It exposes parser output through ``out`` and
``getOutput()`` and the underlying network's passthrough streams.
`HostParsingNeuralNetwork` offers the same interface with the Python parser
implementations from this package. `ExtendedNeuralNetwork` adds input resizing
and optional coordinate remapping around ``ParsingNeuralNetwork``.

`ParserGenerator` creates parsers from NN Archive head metadata. Its
``hostOnly=False`` default selects native DepthAI parsers; ``hostOnly=True``
selects this package's implementations. This flag selects the implementation,
not the device on which it executes. Native parser placement also depends on
DepthAI and the device platform. On RVC2, this package reports that
non-detection native parsers and detection parsers with mask outputs run on
the host. Most native parsers are in ``dai.beta.node``; detection and
segmentation parsers are in ``dai.node``.

``ParsingNeuralNetwork`` accepts an archive object, a Model Zoo reference, or a
``dai.NNModelDescription`` through ``nnSource``. For a local archive, construct
``dai.NNArchive(path)`` first. ``ExtendedNeuralNetwork`` accepts the same source
forms through ``nnSource``. See `ParsingNeuralNetwork.build` and
`ExtendedNeuralNetwork.build` for accepted inputs and configuration.

Example:
    Build native segmentation inference from a Model Zoo reference::

        import depthai as dai
        from depthai_nodes.node import ParsingNeuralNetwork

        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build()
            nn = pipeline.create(ParsingNeuralNetwork).build(
                camera,
                nnSource="luxonis/mediapipe-selfie-segmentation:256x144",
            )
            results = nn.out.createOutputQueue()
            pipeline.start()
            mask = results.get()

    Replace ``ParsingNeuralNetwork`` with ``HostParsingNeuralNetwork`` to
    select the host parser implementation.

Host parser catalogue
=====================

Host parsers consume ``dai.NNData`` and emit native DepthAI messages. Model
metadata must match the parser's expected tensor names and shapes. All parsers
in this catalogue can be imported from either ``depthai_nodes.node`` or
``depthai_nodes.node.parsers``.

Object detection
----------------

* `DetectionParser`: generic bounding-box and score tensors.
* `YOLOExtendedParser`: supported YOLO detection, pose, and instance
  segmentation variants.
* `YuNetParser` and `SCRFDParser`: face detection.
* `MPPalmDetectionParser`: MediaPipe palm detection.
* `PPTextDetectionParser`: PaddleOCR text detection.
* `RFDETRParser`: RF-DETR detections and optional
  instance masks.

These parsers emit ``dai.ImgDetections`` with the fields supported by the model.

Classification and segmentation
-------------------------------

* `ClassificationParser`: class names and scores in ``dai.beta.Classifications``.
* `ClassificationSequenceParser`: per-step classifications, including text
  recognition, in ``dai.beta.Classifications``.
* `SegmentationParser`: semantic masks in ``dai.SegmentationMask``.
* `FastSAMParser`: prompted segmentation in ``dai.SegmentationMask``.

Keypoints and feature matching
------------------------------

* `KeypointParser`: 2D or 3D keypoint tensors.
* `HRNetParser`: body keypoints from heatmaps.
* `SuperAnimalParser`: animal landmarks from heatmaps.

These keypoint parsers emit ``dai.beta.Keypoints``. `XFeatMonoParser` emits
``dai.TrackedFeatures`` from one source; `XFeatStereoParser` matches features
between two sources and emits paired ``dai.TrackedFeatures``.

Other outputs
-------------

* `LaneDetectionParser`: UFLD lane points in ``dai.beta.Clusters``.
* `MLSDParser`: line detections in ``dai.beta.Lines``.
* `EmbeddingsParser`: passes through ``dai.NNData`` embedding outputs.
* `RegressionParser`: regression values in ``dai.beta.Predictions``.
* `MapOutputParser`: numeric maps, including depth estimates, in ``dai.beta.Map2D``.
* `ImageOutputParser`: image-to-image results in ``dai.ImgFrame``.

Utility nodes
=============

Image and depth processing
--------------------------

* `ApplyColormap`: colorizes 2D maps and masks into ``dai.ImgFrame`` messages.
* `ApplyDepthColormap`: uses percentile normalization for depth or disparity;
  non-positive values are excluded from normalization and displayed as black.
* `DepthMerger`: combines 2D detections and aligned depth into
  ``dai.SpatialImgDetections`` using device calibration.
* `HostSpatialsCalc`: computes spatial coordinates from depth regions on the host.
* `FrameCropper`: crops and resizes frames using detections or groups of
  ``dai.ImageManipConfig`` messages, emitting one frame per crop.
* `Tiling`: produces tiled ``dai.ImageManipConfig`` groups and accepts runtime
  configuration changes.
* `ImgFrameOverlay`: blends two ``dai.ImgFrame`` streams.

Detection processing
--------------------

* `CoordinatesMapper`: maps supported messages from crops or tiles into a
  target image transformation, including map and mask pixels.
* `ImgDetectionsFilter`: filters native detections by label, confidence, or
  area, with optional suppression, sorting, and count limits.
* `InstanceToSemanticMask`: replaces instance IDs with detection class labels.

Example:
    With an existing pipeline and detection-producing ``nn.out`` stream::

        from depthai_nodes.node import ImgDetectionsFilter

        filtered = pipeline.create(ImgDetectionsFilter).build(nn.out)
        filtered.keepLabels([0]).minConfidence(0.5).sortByConfidence().takeFirstK(10)

    This keeps class 0 detections, orders them by descending confidence, and
    retains at most ten. ``rejectLabels([0])`` excludes class 0 instead.

Message handling and uploads
----------------------------

* `GatherData`: collects a configurable number of timestamp-matched messages
  per reference into `depthai_nodes.message.GatheredData`. The default count
  is the number of detections in the reference message.
* `MessageCollector`: batches messages within half a frame interval into
  `depthai_nodes.message.Collection`, emitting when a newer timestamp falls outside the
  matching window.
* `SnapsUploader`: accepts `depthai_nodes.message.SnapData` and delegates uploads
  and optional offline caching to the native EventsManager.

Extending the package
=====================

`BaseHostNode` supplies platform-specific image formats and logging for
synchronized ``process()`` callbacks. `BaseThreadedHostNode` provides the same
setup for nodes with their own ``run()`` loop. `BaseParser` defines the common
NNData input and parsed-message output interface.

Parser implementations separate tensor extraction, array computation, and
message emission. Document tensor layouts and coordinate conventions on the
processing helper and preserve input metadata when emitting results.
"""

from .apply_colormap import ApplyColormap
from .apply_depth_colormap import ApplyDepthColormap
from .base_host_node import BaseHostNode
from .base_threaded_host_node import BaseThreadedHostNode
from .coordinates_mapper import CoordinatesMapper
from .depth_merger import DepthMerger
from .extended_neural_network import ExtendedNeuralNetwork
from .frame_cropper import FrameCropper
from .gather_data import GatherData
from .host_parsing_neural_network import HostParsingNeuralNetwork
from .host_spatials_calc import HostSpatialsCalc
from .img_detections_filter import ImgDetectionsFilter
from .img_frame_overlay import ImgFrameOverlay
from .instance_to_semantic_mask import InstanceToSemanticMask
from .message_collector import MessageCollector
from .parser_generator import ParserGenerator
from .parsers.base_parser import BaseParser
from .parsers.classification import ClassificationParser
from .parsers.classification_sequence import ClassificationSequenceParser
from .parsers.detection import DetectionParser
from .parsers.embeddings import EmbeddingsParser
from .parsers.fastsam import FastSAMParser
from .parsers.hrnet import HRNetParser
from .parsers.image_output import ImageOutputParser
from .parsers.keypoints import KeypointParser
from .parsers.lane_detection import LaneDetectionParser
from .parsers.map_output import MapOutputParser
from .parsers.mediapipe_palm_detection import MPPalmDetectionParser
from .parsers.mlsd import MLSDParser
from .parsers.ppdet import PPTextDetectionParser
from .parsers.regression import RegressionParser
from .parsers.rf_detr import RFDETRParser
from .parsers.scrfd import SCRFDParser
from .parsers.segmentation import SegmentationParser
from .parsers.superanimal_landmarker import SuperAnimalParser
from .parsers.xfeat import XFeatMonoParser, XFeatStereoParser
from .parsers.yolo import YOLOExtendedParser
from .parsers.yunet import YuNetParser
from .parsing_neural_network import ParsingNeuralNetwork
from .snaps_uploader import SnapsUploader
from .tiling import Tiling

__all__ = [
    "ApplyColormap",
    "ApplyDepthColormap",
    "CoordinatesMapper",
    "DepthMerger",
    "ExtendedNeuralNetwork",
    "Tiling",
    "ParserGenerator",
    "ParsingNeuralNetwork",
    "HostParsingNeuralNetwork",
    "HostSpatialsCalc",
    "ImageOutputParser",
    "YuNetParser",
    "KeypointParser",
    "HRNetParser",
    "SuperAnimalParser",
    "RFDETRParser",
    "MPPalmDetectionParser",
    "SCRFDParser",
    "SegmentationParser",
    "MLSDParser",
    "XFeatMonoParser",
    "XFeatStereoParser",
    "ClassificationParser",
    "YOLOExtendedParser",
    "FastSAMParser",
    "RegressionParser",
    "PPTextDetectionParser",
    "MapOutputParser",
    "ClassificationSequenceParser",
    "LaneDetectionParser",
    "BaseParser",
    "DetectionParser",
    "EmbeddingsParser",
    "GatherData",
    "ImgFrameOverlay",
    "ImgDetectionsFilter",
    "MessageCollector",
    "SnapsUploader",
    "BaseHostNode",
    "InstanceToSemanticMask",
    "FrameCropper",
    "BaseThreadedHostNode",
]

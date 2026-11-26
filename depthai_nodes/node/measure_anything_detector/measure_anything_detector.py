"""One frozen detection and visibility call per calibrated source scene."""

from datetime import timedelta

import depthai as dai
import numpy as np

from depthai_nodes.algorithms import measure_anything as algorithm
from depthai_nodes.message.measure_anything import (
    MeasureAnythingDetection,
    MeasureAnythingDetections,
    MeasureAnythingDetectorConfig,
    _MeasureAnythingDecision,
)
from depthai_nodes.message.support_plane import SupportPlane

from .scene import (
    SceneInputError,
    _check_cloud_configuration,
    _source_timing,
    _upward_support_plane,
    prepare_scene,
)


class MeasureAnythingDetector(dai.node.ThreadedHostNode):
    """Find objects eligible for measurement in calibrated point-cloud scenes.

    Install ``depthai-nodes[measure-anything]`` for the bundled algorithm's
    dependencies. The published accuracy applies to its evaluated parcel and
    camera profile. The packaged ``visualization.html`` in
    ``depthai_nodes/algorithms/measure_anything/implementation/`` explains the
    algorithm, measured examples, accuracy and performance scopes.

    Connect a millimeter XYZ point cloud and calibrated color, left and right
    images through ``build()``. A host ``Sync`` node matches the four streams by
    device timestamp with a span below 10 ms. They must share a device clock and
    capture the same scene interval; sequence counters may differ. Incomplete
    groups are skipped. The images need not be the original cloud-source frames.

    The evaluated image profile is unrectified color at 2000x1500 and unrectified
    left/right at 1280x800, preserving the native field of view and orientation.
    Color accepts NV12 or native 8-bit RGB/BGR formats. Mono views accept GRAY8
    or RAW8. The first configured scene with valid message types checks sizes
    and logs one warning with expected and received values if they differ.
    Processing continues without resizing, rectification or threshold scaling.
    Other profiles have no equivalent accuracy guarantee. Matching dimensions
    alone do not establish matching geometry. NeuralDepth at 768x480 is the
    reference depth profile; this node receives XYZ and cannot check that grid.

    Each image's ``ImgTransformation`` must describe its delivered pixels,
    including intrinsics, distortion, crop/scale and camera pose. All inputs
    must identify the same calibrated reference frame. XYZ and ``SupportPlane``
    use that frame and millimeters. Native PointCloud producers use DEFAULT
    coordinates, calibrated translation and an identity custom transform.
    Keep the cloud coordinate configuration fixed while running. The reference
    camera must be above the support surface. Either plane-normal sign is
    accepted; a plane through the reference origin is invalid.

    The initial ``SupportPlane`` in ``build()`` is optional. While unset, this
    node consumes synchronized groups without scene decoding, algorithm calls
    or outputs. A ``MeasureAnythingDetectorConfig`` on ``inputConfig`` activates
    processing without rebuilding the pipeline. One queued update is applied
    before each scene when available; the active state otherwise persists.
    An update received during processing applies to a later scene. Use a blocking
    producer queue. A full queue blocks its sender, so send bursts from another
    thread if the caller must also supply scene messages.

    ``out`` emits a ``MeasureAnythingDetections`` batch with eligible objects'
    independent, immutable float32 point arrays in selected source-point order.
    It carries the exact upward plane snapshot, frame identity, source timing,
    algorithm identity and proposal decisions. Candidate IDs are frame-local
    proposal indices. A successful empty frame differs from a failed frame.
    The dimensioner consumes the batch's plane rather than a later config.
    There is no tracking, temporal smoothing or public algorithm-settings API.
    Nested Python values remain in one host runtime and do not serialize through
    XLink or RemoteConnection.

    Example:
        With existing outputs carrying native calibration metadata::

            from depthai_nodes.message import MeasureAnythingDetectorConfig, SupportPlane
            from depthai_nodes.node import MeasureAnythingDetector, MeasureAnythingDimensioner

            detector = pipeline.create(MeasureAnythingDetector).build(
                pointCloud, color, left, right
            )
            configQueue = detector.inputConfig.createInputQueue(blocking=True)
            dimensioner = pipeline.create(MeasureAnythingDimensioner).build(detector.out)
            results = dimensioner.out.createOutputQueue()

            pipeline.start()
            supportPlane = SupportPlane.fromJsonFile("support_plane.json")
            configQueue.send(MeasureAnythingDetectorConfig(supportPlane=supportPlane))
    """

    algorithm_id = algorithm.ALGORITHM_ID
    _SYNC_TOLERANCE = timedelta(milliseconds=10)

    def __init__(self) -> None:
        super().__init__()
        self._sceneInput = self.createInput("scene")
        self._inputConfig = self.createInput("config")
        self._out = self.createOutput("out")
        self._components = None
        self._config: MeasureAnythingDetectorConfig | None = None
        self._resolutionsChecked = False

    @property
    def inputConfig(self) -> dai.Node.Input:
        """Optional runtime updates, applied in queue order between scenes."""
        return self._inputConfig

    @property
    def out(self) -> dai.Node.Output:
        """One eligible-object batch or failure per processed synchronized group."""
        return self._out

    def build(
        self,
        pointCloud: dai.Node.Output,
        color: dai.Node.Output,
        left: dai.Node.Output,
        right: dai.Node.Output,
        supportPlane: SupportPlane | None = None,
    ) -> "MeasureAnythingDetector":
        """Connect scene streams with an optional initial support plane.

        Args:
            pointCloud: ``dai.PointCloudData`` with finite millimeter XYZ in
                the common calibration reference frame. Native PointCloud
                producers must use DEFAULT coordinates, calibrated translation,
                and an identity custom transform. Organized clouds are optional.
            color: Unrectified 2000x1500 NV12 or native 8-bit RGB/BGR images of
                the same capture interval.
            left: Unrectified 1280x800 calibrated mono images of the same scene
                and capture interval.
            right: Unrectified 1280x800 calibrated mono images of the same scene
                and capture interval.
            supportPlane: Optional initial plane in the point cloud's reference frame.
                The reference camera must be above the surface. Either normal
                sign is accepted. If omitted, incoming groups are consumed without
                processing or output until ``inputConfig`` supplies a plane.

        Returns:
            This connected node.

        Raises:
            RuntimeError: If already built.
            TypeError: If ``supportPlane`` is neither a ``SupportPlane`` nor ``None``.
            ValueError: If a native PointCloud producer has incompatible coordinates
                or units. Its coordinate configuration must remain fixed at runtime.
            ImportError: If the measure-anything extra dependencies are missing.
        """
        if self._components is not None:
            raise RuntimeError("MeasureAnythingDetector is already built")
        config = (
            MeasureAnythingDetectorConfig(supportPlane=supportPlane)
            if supportPlane is not None
            else None
        )
        producer = pointCloud.getParent()
        if isinstance(producer, dai.node.PointCloud):
            _check_cloud_configuration(producer.initialConfig)
        components = algorithm.create_components()
        sync = self.createSubnode(dai.node.Sync)
        sync.setRunOnHost(True)
        sync.setSyncThreshold(self._SYNC_TOLERANCE)
        sync.setSyncAttempts(-1)
        sync.setTimestampSource(dai.node.Sync.TimestampSource.DEVICE)
        for streamName, output in (
            ("pointCloud", pointCloud),
            ("color", color),
            ("left", left),
            ("right", right),
        ):
            output.link(sync.inputs[streamName])
        sync.out.link(self._sceneInput)
        self._components = components
        self._config = config
        return self

    def _update_config(self, config: MeasureAnythingDetectorConfig) -> None:
        if not isinstance(config, MeasureAnythingDetectorConfig):
            self.warn("Ignoring config input: expected MeasureAnythingDetectorConfig")
            return
        self._config = config

    def process(
        self,
        pointCloud: dai.Buffer,
        color: dai.Buffer,
        left: dai.Buffer,
        right: dai.Buffer,
    ) -> None:
        """Process one synchronized scene using one active plane snapshot.

        Without a configured support plane, return without processing or output. Input
        datatype checks happen here. Private conversion helpers receive validated scene
        messages. Invalid inputs, plane geometry, metadata, or callback results produce
        a failed batch. The source point cloud's native timestamps and sequence number
        are preserved on every emitted result.
        """
        if self._components is None:
            raise RuntimeError("Build MeasureAnythingDetector before processing")
        if self._config is None:
            return
        supportPlane = None
        frame = "unknown"
        sources = tuple(
            _source_timing(streamName, message)
            for streamName, message in (
                ("pointCloud", pointCloud),
                ("color", color),
                ("left", left),
                ("right", right),
            )
        )
        try:
            if not isinstance(pointCloud, dai.PointCloudData):
                raise SceneInputError("invalid_point_cloud")
            if any(
                not isinstance(imgFrame, dai.ImgFrame)
                for imgFrame in (color, left, right)
            ):
                raise SceneInputError("invalid_source_image")
            if not self._resolutionsChecked:
                self._resolutionsChecked = True
                imageSizes = tuple(
                    (imgFrame.getWidth(), imgFrame.getHeight())
                    for imgFrame in (color, left, right)
                )
                if imageSizes != ((2000, 1500), (1280, 800), (1280, 800)):
                    self.warn(
                        "MeasureAnythingDetector expects unrectified RGB 2000x1500, "
                        "left 1280x800, and right 1280x800 images. "
                        f"Received RGB {imageSizes[0][0]}x{imageSizes[0][1]}, "
                        f"left {imageSizes[1][0]}x{imageSizes[1][1]}, "
                        f"and right {imageSizes[2][0]}x{imageSizes[2][1]}. "
                        "Processing will continue; other resolutions may change "
                        "detection eligibility decisions."
                    )
            supportPlane = _upward_support_plane(self._config.supportPlane)
            scene = prepare_scene(pointCloud, color, left, right, supportPlane)
            frame = scene.cloud.frame
            candidates = self._components.find_candidates(scene)
            if not isinstance(candidates, list):
                raise ValueError("find_candidates must return a list")
            detections, decisions = [], []
            for number, candidate in enumerate(candidates):
                if not isinstance(
                    candidate, algorithm.CandidateResult
                ) or not isinstance(candidate.segment, algorithm.Segment):
                    raise ValueError("Invalid candidate result")
                indices = candidate.segment.point_indices
                if (
                    not isinstance(indices, np.ndarray)
                    or indices.ndim != 1
                    or indices.dtype != np.int32
                    or indices.size == 0
                    or indices.min() < 0
                    or indices.max() >= len(scene.cloud.xyz_mm)
                    or (
                        not np.all(indices[1:] > indices[:-1])
                        and np.unique(indices).size != indices.size
                    )
                ):
                    raise ValueError("Invalid candidate segment indices")
                decision = _MeasureAnythingDecision(
                    candidate_id=number, decision=candidate.decision
                )
                decisions.append(decision)
                if decision.decision == "eligible":
                    points = np.take(scene.cloud.xyz_mm, indices, axis=0)
                    detections.append(
                        MeasureAnythingDetection._from_owned(number, points)
                    )
            result = MeasureAnythingDetections(
                items=tuple(detections),
                decisions=tuple(decisions),
                frame=frame,
                plane=supportPlane,
                sources=sources,
                algorithm_id=self.algorithm_id,
            )
        except Exception as exc:
            result = MeasureAnythingDetections(
                frame=frame,
                plane=supportPlane,
                sources=sources,
                algorithm_id=self.algorithm_id,
                outcome=MeasureAnythingDetections.Outcome.FAILED,
                failure_code=exc.code
                if isinstance(exc, SceneInputError)
                else "detection_failed",
            )
        self.out.send(result)

    def run(self) -> None:
        """Receive synchronized scenes and apply an available configuration update."""
        while self.isRunning():
            group = self._sceneInput.get()
            config = self.inputConfig.tryGet()
            if config is not None:
                self._update_config(config)
            self.process(
                group["pointCloud"],
                group["color"],
                group["left"],
                group["right"],
            )

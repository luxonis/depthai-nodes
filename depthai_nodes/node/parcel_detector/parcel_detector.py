"""One frozen detection and visibility call per calibrated source scene."""

from datetime import timedelta

import depthai as dai
import numpy as np

from depthai_nodes.algorithms import parcel as algorithm
from depthai_nodes.message.parcel import (
    Parcel,
    ParcelDetections,
    ParcelDetectorConfig,
    _ParcelDecision,
)
from depthai_nodes.message.support_plane import SupportPlane

from .scene import (
    SceneInputError,
    _check_cloud_configuration,
    _source_timing,
    _upward_support_plane,
    prepare_scene,
)


class ParcelDetector(dai.node.ThreadedHostNode):
    """Detect eligible parcels using the single bundled fast baseline.

    A host ``Sync`` node matches the point cloud and three image streams by
    device timestamp within 10 ms. Incomplete groups produce no callback.
    Image sequence counters may differ from the point cloud's counter.

    The evaluated profile uses unrectified color at 2000x1500 and left/right
    images at 1280x800. The first scene with valid message types checks these
    sizes and logs one warning if they differ. Processing continues unchanged.

    ``inputConfig`` accepts ``ParcelDetectorConfig`` messages. One queued
    configuration is applied before each synchronized scene, when available.
    The active configuration persists until another update arrives.
    Its support-plane snapshot travels with the eligible parcels through
    measurement. Updates received during processing apply to a later scene.

    ``out`` emits same-host ``ParcelDetections`` with independent read-only
    parcel arrays. Candidate IDs are frame-local proposal indices. There is no
    tracking, temporal aggregation, or frozen-algorithm configuration API.
    """

    algorithm_id = algorithm.ALGORITHM_ID
    _SYNC_TOLERANCE = timedelta(milliseconds=10)

    def __init__(self) -> None:
        super().__init__()
        self._sceneInput = self.createInput("scene")
        self._inputConfig = self.createInput("config")
        self._out = self.createOutput("out")
        self._components = None
        self._config: ParcelDetectorConfig | None = None
        self._resolutionsChecked = False

    @property
    def inputConfig(self) -> dai.Node.Input:
        """Optional runtime updates, applied in queue order between scenes."""
        return self._inputConfig

    @property
    def out(self) -> dai.Node.Output:
        """One eligible-parcel batch or failure per processed synchronized group."""
        return self._out

    def build(
        self,
        pointCloud: dai.Node.Output,
        color: dai.Node.Output,
        left: dai.Node.Output,
        right: dai.Node.Output,
        supportPlane: SupportPlane,
    ) -> "ParcelDetector":
        """Connect the scene streams and initialize the active support plane.

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
            supportPlane: Initial plane in the point cloud's reference frame.
                The reference camera must be above the surface. Either normal
                sign is accepted.

        Returns:
            This configured node.

        Raises:
            RuntimeError: If already built.
            TypeError: If ``supportPlane`` is not a ``SupportPlane``.
            ValueError: If a native PointCloud producer has incompatible coordinates
                or units. Its coordinate configuration must remain fixed at runtime.
            ImportError: If the parcel extra dependencies are missing.
        """
        if self._components is not None:
            raise RuntimeError("ParcelDetector is already built")
        config = ParcelDetectorConfig(supportPlane=supportPlane)
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

    def _update_config(self, config: ParcelDetectorConfig) -> None:
        if not isinstance(config, ParcelDetectorConfig):
            self.warn("Ignoring config input: expected ParcelDetectorConfig")
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

        Input datatype checks happen here. Private conversion helpers receive validated
        scene messages. Invalid inputs, plane geometry, metadata, or callback results
        produce a failed batch. The source point cloud's native timestamps and sequence
        number are preserved on every emitted result.
        """
        if self._components is None or self._config is None:
            raise RuntimeError("Build ParcelDetector before processing")
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
                        "ParcelDetector expects unrectified RGB 2000x1500, "
                        "left 1280x800, and right 1280x800 images. "
                        f"Received RGB {imageSizes[0][0]}x{imageSizes[0][1]}, "
                        f"left {imageSizes[1][0]}x{imageSizes[1][1]}, "
                        f"and right {imageSizes[2][0]}x{imageSizes[2][1]}. "
                        "Processing will continue; other resolutions may change "
                        "parcel eligibility decisions."
                    )
            supportPlane = _upward_support_plane(self._config.supportPlane)
            scene = prepare_scene(pointCloud, color, left, right, supportPlane)
            frame = scene.cloud.frame
            candidates = self._components.find_candidates(scene)
            if not isinstance(candidates, list):
                raise ValueError("find_candidates must return a list")
            parcels, decisions = [], []
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
                decision = _ParcelDecision(
                    candidate_id=number, decision=candidate.decision
                )
                decisions.append(decision)
                if decision.decision == "eligible":
                    points = np.take(scene.cloud.xyz_mm, indices, axis=0)
                    parcels.append(Parcel._from_owned(number, points))
            result = ParcelDetections(
                items=tuple(parcels),
                decisions=tuple(decisions),
                frame=frame,
                plane=supportPlane,
                sources=sources,
                algorithm_id=self.algorithm_id,
            )
        except Exception as exc:
            result = ParcelDetections(
                frame=frame,
                plane=supportPlane,
                sources=sources,
                algorithm_id=self.algorithm_id,
                outcome=ParcelDetections.Outcome.FAILED,
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

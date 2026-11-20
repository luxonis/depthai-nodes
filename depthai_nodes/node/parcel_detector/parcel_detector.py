"""One frozen detection and visibility call per calibrated source scene."""

from datetime import timedelta

import depthai as dai
import numpy as np

from depthai_nodes.algorithms import parcel as algorithm
from depthai_nodes.message.parcel import (
    Parcel,
    ParcelDetections,
    ParcelRigCalibration,
    _ParcelDecision,
)

from .scene import SceneInputError, prepare_scene, source_timing, upward_plane


class ParcelDetector(dai.node.HostNode):
    """Detect eligible parcels using the single bundled fast baseline.

    DepthAI synchronizes the five input streams before calling ``process``.
    Each synchronized group must have matching sequence numbers and device
    timestamps within 10 ms. Incomplete groups do not invoke the callback.
    A supplied plane applies only to its matching cloud sequence, never to a
    newer frame.

    ``out`` emits same-host ``ParcelDetections`` with independent read-only
    parcel arrays. Candidate IDs are frame-local proposal indices. There is no
    tracking, temporal aggregation, or algorithm configuration API.
    """

    algorithm_id = algorithm.ALGORITHM_ID
    _SOURCE_NAMES = ("color", "left", "right", "support_plane")
    _SYNC_TOLERANCE = timedelta(milliseconds=10)

    def __init__(self) -> None:
        super().__init__()
        self._calibration: ParcelRigCalibration | None = None
        self._components = None

    def build(
        self,
        pointCloud: dai.Node.Output,
        color: dai.Node.Output,
        left: dai.Node.Output,
        right: dai.Node.Output,
        supportPlane: dai.Node.Output,
        calibration: ParcelRigCalibration,
    ) -> "ParcelDetector":
        """Connect the complete scene and bind the frozen algorithm.

        Args:
            pointCloud: ``dai.PointCloudData`` with finite XYZ in millimeters,
                expressed in ``calibration.frame``. Organized clouds are optional.
            color: NV12 or native 8-bit RGB/BGR image stream.
            left: GRAY8 left image stream.
            right: GRAY8 right image stream.
            supportPlane: Same-host ``ParcelPlane`` stream, stamped for each cloud.
            calibration: Exact delivered camera grids, transforms, frame, and gravity.

        Returns:
            This configured node.

        Raises:
            RuntimeError: If already built.
            TypeError: If calibration has the wrong type.
            ImportError: If the parcel extra dependencies are missing.
        """
        if self._components is not None:
            raise RuntimeError("ParcelDetector is already built")
        if not isinstance(calibration, ParcelRigCalibration):
            raise TypeError("calibration must be ParcelRigCalibration")
        components = algorithm.create_components()
        self.link_args(pointCloud, color, left, right, supportPlane)
        self._components = components
        self._calibration = calibration
        return self

    def process(
        self,
        pointCloud: dai.Buffer,
        color: dai.Buffer,
        left: dai.Buffer,
        right: dai.Buffer,
        supportPlane: dai.Buffer,
    ) -> None:
        """Process one group synchronized by DepthAI and emit its outcome.

        Missing inputs, source mismatch, invalid scene data, and callback exceptions
        produce failed batches. Only a completed callback can produce a successful empty
        batch. The cloud's native timing is always preserved.
        """
        if self._components is None or self._calibration is None:
            raise RuntimeError("Build ParcelDetector before processing")
        if not isinstance(pointCloud, dai.Buffer):
            raise TypeError("pointCloud must be a DepthAI Buffer")
        inputs = (color, left, right, supportPlane)
        source_messages = (
            ("point_cloud", pointCloud),
            *zip(self._SOURCE_NAMES, inputs),
        )
        sources = tuple(
            source_timing(name, message)
            for name, message in source_messages
            if isinstance(message, dai.Buffer)
        )
        plane = None
        try:
            plane = upward_plane(supportPlane, self._calibration)
            if any(not isinstance(message, dai.Buffer) for message in inputs):
                raise SceneInputError("missing_source_image")
            if any(
                message.getSequenceNum() != pointCloud.getSequenceNum()
                or abs(message.getTimestampDevice() - pointCloud.getTimestampDevice())
                > self._SYNC_TOLERANCE
                for message in inputs
            ):
                raise SceneInputError("unsynchronized_sources")
            scene = prepare_scene(
                pointCloud, (color, left, right), plane, self._calibration
            )
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
                frame=self._calibration.frame,
                plane=plane,
                sources=sources,
                algorithm_id=self.algorithm_id,
            )
        except Exception as exc:
            result = ParcelDetections(
                frame=self._calibration.frame,
                plane=plane,
                sources=sources,
                algorithm_id=self.algorithm_id,
                outcome=ParcelDetections.Outcome.FAILED,
                failure_code=exc.code
                if isinstance(exc, SceneInputError)
                else "detection_failed",
            )
        self.out.send(result)

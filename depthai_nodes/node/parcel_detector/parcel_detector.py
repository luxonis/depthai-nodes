"""One frozen detection and visibility call per calibrated source scene."""

from __future__ import annotations

import time
from datetime import timedelta

import depthai as dai
import numpy as np

from depthai_nodes.algorithms import parcel as algorithm
from depthai_nodes.message.parcel import (
    Parcel,
    ParcelDecision,
    ParcelDetections,
    ParcelFrameOutcome,
    ParcelRigCalibration,
)

from .scene import SceneInputError, prepare_scene, source_timing, upward_plane


class ParcelDetector(dai.node.ThreadedHostNode):
    """Detect eligible parcels using the single bundled fast baseline.

    Inputs are ordered streams with the same sequence number and device
    timestamps within 10 ms. Each cloud triggers one result. Required images
    and a per-frame ``ParcelPlane`` may arrive up to 250 ms after the cloud is
    dequeued. A missing or mismatched source emits a failed frame. A supplied
    plane applies only to its matching cloud sequence, never to a newer frame.

    ``out`` emits same-host ``ParcelDetections`` with independent read-only
    parcel arrays. Candidate IDs are frame-local proposal indices. There is no
    tracking, temporal aggregation, or algorithm configuration API.
    """

    algorithm_id = algorithm.ALGORITHM_ID
    _SOURCE_NAMES = ("color", "left", "right", "support_plane")
    _SYNC_TOLERANCE = timedelta(milliseconds=10)
    _INPUT_TIMEOUT_SECONDS = 0.250
    _MAX_PENDING = 8

    def __init__(self) -> None:
        super().__init__()
        self._point_cloud_input = self.createInput("point_cloud", queueSize=8)
        self._scene_inputs = {
            name: self.createInput(name, queueSize=8) for name in self._SOURCE_NAMES
        }
        self._out = self.createOutput("out")
        self._calibration: ParcelRigCalibration | None = None
        self._components = None

    @property
    def out(self) -> dai.Node.Output:
        """One eligible-parcel batch or failed-frame result per source cloud."""
        return self._out

    def build(
        self,
        pointCloud: dai.Node.Output,
        color: dai.Node.Output,
        left: dai.Node.Output,
        right: dai.Node.Output,
        supportPlane: dai.Node.Output,
        calibration: ParcelRigCalibration,
    ) -> ParcelDetector:
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
        pointCloud.link(self._point_cloud_input)
        for name, output in zip(self._SOURCE_NAMES, (color, left, right, supportPlane)):
            output.link(self._scene_inputs[name])
        self._components = components
        self._calibration = calibration
        return self

    def process(
        self,
        pointCloud: dai.Buffer,
        color: dai.Buffer | None = None,
        left: dai.Buffer | None = None,
        right: dai.Buffer | None = None,
        supportPlane: dai.Buffer | None = None,
    ) -> None:
        """Process one explicitly matched source frame and emit its outcome.

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
                decision = ParcelDecision(
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
                outcome=ParcelFrameOutcome.FAILED,
                failure_code=exc.code
                if isinstance(exc, SceneInputError)
                else "detection_failed",
            )
        self.out.send(result)

    def run(self) -> None:
        """Match each cloud to its required sources, with bounded incomplete frames."""
        pending: dict[str, dict[int, dai.Buffer]] = {
            name: {} for name in self._SOURCE_NAMES
        }
        while self.isRunning():
            cloud = self._point_cloud_input.get()
            if cloud is None:
                continue
            sequence = cloud.getSequenceNum()
            deadline = time.monotonic() + self._INPUT_TIMEOUT_SECONDS
            while self.isRunning():
                for name, input_queue in self._scene_inputs.items():
                    entries = pending[name]
                    for old in [s for s in entries if s < sequence]:
                        del entries[old]
                    while len(entries) < self._MAX_PENDING:
                        message = input_queue.tryGet()
                        if message is None:
                            break
                        if message.getSequenceNum() >= sequence:
                            entries[message.getSequenceNum()] = message
                if all(sequence in pending[name] for name in self._SOURCE_NAMES):
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.001)
            if self.isRunning():
                self.process(
                    cloud,
                    *(pending[name].pop(sequence, None) for name in self._SOURCE_NAMES),
                )

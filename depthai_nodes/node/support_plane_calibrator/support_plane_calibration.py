"""Search for and verify support planes in a fixed installation."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from weakref import WeakSet, ref

import depthai as dai
import numpy as np

from depthai_nodes.logging import get_logger
from depthai_nodes.message.auto_support_plane_config import _AutoConfigUpdate
from depthai_nodes.message.plane_calibration_config import (
    PlaneCalibrationConfig,
)
from depthai_nodes.message.plane_calibration_config import (
    SupportPlaneCalibrationConfig as SupportPlaneCalibrationConfig,
)
from depthai_nodes.message.support_plane_calibration import (
    SupportPlaneCalibrationControl,
    SupportPlaneCandidateBatch,
    SupportPlaneVerificationRequest,
    SupportPlaneVerificationResult,
)

from ._plane_assessment import assess_plane
from .support_plane_calibrator import (
    IMU_MAX_AGE_SECONDS,
    MIN_GRAVITY_NORM,
    _check_size,
)
from .support_plane_fitter import SupportPlaneFitter


@dataclass(frozen=True)
class _GravitySample:
    direction: tuple[float, float, float]
    timestamp_device: timedelta | None
    timestamp_host: timedelta | None
    timestamp_system: datetime | None
    arrival: float


@dataclass
class _Search:
    last_fit_start: float = -math.inf
    # Recognize resubmitted messages without retaining their point buffers.
    fitted_clouds: WeakSet[dai.PointCloudData] = field(default_factory=WeakSet)


def _latest_gravity(imu: dai.IMUData, arrival: float) -> _GravitySample | None:
    for packet in reversed(imu.packets):
        report = packet.acceleroMeter
        norm = math.hypot(report.x, report.y, report.z)
        if math.isfinite(norm) and norm >= MIN_GRAVITY_NORM:
            return _GravitySample(
                (-report.x / norm, -report.y / norm, -report.z / norm),
                report.getTimestampDevice() or None,
                report.getTimestamp() or None,
                report.getTimestampSystem(),
                arrival,
            )
    return None


class SupportPlaneCalibration(dai.node.ThreadedHostNode):
    """Fit candidate planes and assess supplied equations on an organized cloud.

    Send START or STOP to ``inputControl``. Each fit emits a batch on
    ``candidatesOutput``. A batch with an accepted assessment ends the search.
    Send requests to ``inputVerification`` to receive independent, correlated
    replies on ``verificationOutput``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pointCloud = self.createInput(
            name="pointCloud", blocking=False, queueSize=1
        )
        self.imu = self.createInput(name="imu", blocking=False, queueSize=1)
        self.inputControl = self.createInput(name="inputControl")
        self.inputVerification = self.createInput(name="inputVerification")
        self.candidatesOutput = self.createOutput(name="candidatesOutput")
        self.verificationOutput = self.createOutput(name="verificationOutput")
        self._latest_cloud: dai.PointCloudData | None = None
        self._gravity: _GravitySample | None = None
        self._search: _Search | None = None
        self._auto_config_revision = 0
        self._requests: deque[SupportPlaneVerificationRequest] = deque()
        self._fitter: SupportPlaneFitter | None = None
        self._logger = get_logger(type(self).__name__)

    def build(
        self,
        pointCloud: dai.Node.Output,
        imu: dai.Node.Output,
        pointCloudSize: tuple[int, int],
        config: SupportPlaneCalibrationConfig | None = None,
    ) -> SupportPlaneCalibration:
        """Link the millimeter cloud and the IMU output.

        Args:
            pointCloud: Organized point cloud output.
            imu: Accelerometer output with vectors in the point cloud frame.
            pointCloudSize: Organized cloud size as (width, height).
            config: Fit scheduling and acceptance settings.
        """
        if self._fitter is not None:
            raise RuntimeError("SupportPlaneCalibration is already built")
        try:
            import open3d  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SupportPlaneCalibration requires Open3D. Install it with "
                '`pip install "depthai-nodes[box-measurement]"`.'
            ) from exc
        if config is not None and not isinstance(config, SupportPlaneCalibrationConfig):
            raise TypeError("config must be a SupportPlaneCalibrationConfig")
        self._cloud_size = _check_size("pointCloudSize", pointCloudSize)
        self._config = config if config is not None else SupportPlaneCalibrationConfig()
        self._source_stream_id = (
            f"{pointCloud.getParent().id}/{pointCloud.getGroup()}/"
            f"{pointCloud.getName()}"
        )
        node_ref = ref(self)

        def receive_gravity(message: dai.ADatatype) -> None:
            node = node_ref()
            if node is not None and isinstance(message, dai.IMUData):
                gravity = _latest_gravity(message, time.monotonic())
                if gravity is not None:
                    node._gravity = gravity

        pointCloud.link(self.pointCloud)
        imu.link(self.imu)
        # Capture valid reports at ingress before the latest-only queue can replace them.
        self.imu.addCallback(receive_gravity)
        self._fitter = SupportPlaneFitter(self._cloud_size, PlaneCalibrationConfig())
        return self

    def run(self) -> None:
        """Consume sensor and request messages until the pipeline stops."""
        if self._fitter is None:
            raise RuntimeError(
                "Build SupportPlaneCalibration before starting the pipeline"
            )
        try:
            while self.isRunning():
                self._step()
                time.sleep(0.005)
        except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
            return

    def _step(self) -> None:
        commands = SupportPlaneCalibrationControl.Commands
        for control in self.inputControl.tryGetAll():
            if isinstance(control, _AutoConfigUpdate):
                assert control.config._assessment is not None
                self._config = control.config._assessment
                self._auto_config_revision = control.revision
                self._search = _Search() if control.searching else None
                self._requests.clear()
            elif isinstance(control, SupportPlaneVerificationRequest):
                self._requests.append(control)
            elif isinstance(control, SupportPlaneCalibrationControl):
                if isinstance(control.command, commands.Start) and self._search is None:
                    self._search = _Search()
                elif isinstance(control.command, commands.Stop):
                    self._search = None
        self._requests.extend(
            request
            for request in self.inputVerification.tryGetAll()
            if isinstance(request, SupportPlaneVerificationRequest)
        )
        for cloud in self.pointCloud.tryGetAll():
            if isinstance(cloud, dai.PointCloudData):
                self._latest_cloud = cloud
        cloud, gravity = self._latest_cloud, self._gravity
        now = time.monotonic()
        if (
            cloud is None
            or gravity is None
            or now - gravity.arrival > IMU_MAX_AGE_SECONDS
        ):
            return
        search = self._search
        fit_due = (
            search is not None
            and cloud not in search.fitted_clouds
            and now - search.last_fit_start >= self._config.attempt_interval_sec
        )
        if not self._requests and not fit_due:
            return

        width, height = self._cloud_size
        actual_size = (cloud.getWidth(), cloud.getHeight())
        if actual_size not in ((0, 0), self._cloud_size):
            self._logger.warning(
                "Expected organized cloud of size %s, got %s",
                self._cloud_size,
                actual_size,
            )
            self._latest_cloud = None
            return
        points = np.asarray(cloud.getPoints())
        if points.shape != (width * height, 3):
            self._logger.warning(
                "Expected %d organized cloud points, got shape %s",
                width * height,
                points.shape,
            )
            self._latest_cloud = None
            return
        sample = points.reshape(height, width, 3)[::4, ::4]
        sample = sample[sample[..., 2] > 0]
        while self._requests:
            request = self._requests.popleft()
            result = SupportPlaneVerificationResult.from_assessment(
                assess_plane(request.plane, sample, gravity.direction, self._config),
                request.request_id,
                cloud=cloud,
                gravity_direction=gravity.direction,
                imu_timestamp_device=gravity.timestamp_device,
                source_stream_id=self._source_stream_id,
                imu_timestamp_host=gravity.timestamp_host,
                imu_timestamp_system=gravity.timestamp_system,
            )
            result._auto_config_revision = self._auto_config_revision
            self.verificationOutput.send(result)
        if fit_due and search is not None:
            started = time.monotonic()
            if started - gravity.arrival <= IMU_MAX_AGE_SECONDS:
                search.fitted_clouds.add(cloud)
                search.last_fit_start = started
                self._fit(cloud, points, sample, gravity)

    def _fit(
        self,
        cloud: dai.PointCloudData,
        points: np.ndarray,
        sample: np.ndarray,
        gravity: _GravitySample,
    ) -> None:
        assert self._fitter is not None
        try:
            planes = self._fitter.propose(points, -np.asarray(gravity.direction))
        except (ValueError, RuntimeError) as exc:
            self._logger.warning("Support plane fit failed: %s", exc)
            planes = []
        batch = SupportPlaneCandidateBatch(
            (
                assess_plane(plane, sample, gravity.direction, self._config)
                for plane in planes
            ),
            cloud=cloud,
            gravity_direction=gravity.direction,
            imu_timestamp_device=gravity.timestamp_device,
            source_stream_id=self._source_stream_id,
            imu_timestamp_host=gravity.timestamp_host,
            imu_timestamp_system=gravity.timestamp_system,
        )
        batch._auto_config_revision = self._auto_config_revision
        self.candidatesOutput.send(batch)
        if any(assessment.accepted for assessment in batch.assessments):
            self._search = None

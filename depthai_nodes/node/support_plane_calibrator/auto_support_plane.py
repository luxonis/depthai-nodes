import time
from collections.abc import Callable

import depthai as dai

from depthai_nodes.message.auto_support_plane_result import AutoSupportPlaneResult
from depthai_nodes.message.support_plane import SupportPlane, SupportPlaneCandidate
from depthai_nodes.message.support_plane_calibration import (
    CalibrationOutcome,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
    SupportPlaneCandidateBatch,
    SupportPlaneVerificationRequest,
    SupportPlaneVerificationResult,
)

from .auto_support_plane_config import AutoSupportPlaneConfig
from .plane_calibration_config import PlaneCalibrationConfig
from .support_plane_calibration import SupportPlaneCalibration
from .support_plane_calibrator import SupportPlaneCalibrator

CandidateSelector = Callable[
    [SupportPlaneCalibrationResult], SupportPlaneCandidate | None
]

# The legacy protocol can enqueue Reset, Start, and Verify in one pass.
CONTROL_QUEUE_SIZE = 8


class AutoSupportPlane(dai.node.ThreadedHostNode):
    """Find and hold the support plane from a point cloud and IMU.

    Wraps a `SupportPlaneCalibration`. ``ON_START`` finds a plane once and holds
    it; ``CONTINUOUS`` also checks it about every ``check_interval_sec`` and
    refits after repeated failed checks, keeping the old plane meanwhile. Pass
    the plane held before a pipeline rebuild as ``initialPlane``.

    Attributes:
        out: Output stream of `depthai_nodes.message.AutoSupportPlaneResult`,
            one per found, checked or rejected plane.
        verificationOutput: Output stream of
            `depthai_nodes.message.SupportPlaneVerificationResult`, one per
            ``initialPlane`` gate or plane check, passed or failed.
        candidatesOutput: Output stream of
            `depthai_nodes.message.SupportPlaneCandidateBatch` when using
            `SupportPlaneCalibration`.
    """

    def __init__(self) -> None:
        """Create the internal ports and result output."""
        super().__init__()
        self._calibrator: SupportPlaneCalibration | SupportPlaneCalibrator | None = None
        self._control_output = self.createOutput(name="calibratorControl")
        self._request_output = self.createOutput(name="calibratorVerificationRequest")
        # Blocking and lossless: the wrapper counts on every result arriving.
        self._calibration_input = self.createInput(name="calibrationResult")
        self._verification_input = self.createInput(name="verificationResult")
        self._out = self.createOutput()
        self._config: AutoSupportPlaneConfig | None = None
        self._select_candidate: CandidateSelector = _select_largest_plane
        self._started = False
        # The current plane, kept through a refit.
        self._plane: SupportPlane | None = None
        # Whether a calibration attempt runs (a search or a refit).
        self._searching = False
        self._verify_in_flight = False
        self._request_id = 0
        # Whether the in-flight Verify is the initialPlane gate.
        self._gate_pending = False
        self._next_check_at = 0.0
        self._failed_checks = 0

    @property
    def out(self) -> dai.Node.Output:
        """One `depthai_nodes.message.AutoSupportPlaneResult` per finished cycle.

        Returns:
            ``dai.Node.Output``: The result output.
        """
        return self._out

    @property
    def verificationOutput(self) -> dai.Node.Output:
        """The inner calibrator's plane verifications, one per gate or plane check.

        Use it to watch the angle and on-plane ratio of the held plane, including
        failed plane checks, which send nothing on ``out``.

        Returns:
            ``dai.Node.Output``: The verification result output.
        """
        if self._calibrator is None:
            raise RuntimeError(
                "Build AutoSupportPlane before linking verificationOutput"
            )
        return self._calibrator.verificationOutput

    @property
    def candidatesOutput(self) -> dai.Node.Output:
        """All fitting batches from the inner `SupportPlaneCalibration`."""
        if not isinstance(self._calibrator, SupportPlaneCalibration):
            raise RuntimeError(
                "Build with SupportPlaneCalibrationConfig to link candidatesOutput"
            )
        return self._calibrator.candidatesOutput

    def build(
        self,
        pointCloud: dai.Node.Output,
        imu: dai.Node.Output,
        pointCloudSize: tuple[int, int],
        config: AutoSupportPlaneConfig | None = None,
        initialPlane: SupportPlane | None = None,
        selectCandidate: CandidateSelector | None = None,
    ) -> "AutoSupportPlane":
        """Build the inner calibrator on the point cloud and IMU.

        Args:
            pointCloud: Output emitting organized ``dai.PointCloudData`` in
                millimeters.
            imu: Output emitting ``dai.IMUData`` with accelerometer packets.
            pointCloudSize: Cloud ``(width, height)`` in points.
            config: Mode and calibrator settings; ``None`` uses the defaults.
            initialPlane: Verified on the first cloud with fresh IMU before
                reporting it as held. A rejected plane triggers a search.
            selectCandidate: Optional selector for the legacy calibrator.
                Supplying it without a config selects the legacy calibrator.
                The shared evaluator ranks passing assessments by sampled
                support, then by distance.

        Returns:
            The configured node instance.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If the node is already built.
            TypeError: If ``config`` is not an ``AutoSupportPlaneConfig``,
                ``initialPlane`` is neither ``None`` nor a ``SupportPlane``,
                ``selectCandidate`` is neither ``None`` nor callable, or
                ``pointCloudSize`` is not a pair of integers.
            ValueError: If ``pointCloudSize`` does not have two positive values.
        """
        if self._config is not None:
            raise RuntimeError(
                "AutoSupportPlane is already built; create a new node to change "
                "its configuration"
            )
        if config is None:
            config = (
                AutoSupportPlaneConfig(calibration=PlaneCalibrationConfig())
                if selectCandidate is not None
                else AutoSupportPlaneConfig()
            )
        if not isinstance(config, AutoSupportPlaneConfig):
            raise TypeError(
                "config must be AutoSupportPlaneConfig, got " f"{type(config).__name__}"
            )
        if selectCandidate is not None and not callable(selectCandidate):
            raise TypeError(
                "selectCandidate must be callable or None, got "
                f"{type(selectCandidate).__name__}"
            )
        if initialPlane is not None and not isinstance(initialPlane, SupportPlane):
            raise TypeError(
                "initialPlane must be SupportPlane, got "
                f"{type(initialPlane).__name__}"
            )
        calibration = config.calibration
        if isinstance(calibration, PlaneCalibrationConfig):
            calibrator = self.createSubnode(SupportPlaneCalibrator)
            calibrator.build(pointCloud, imu, pointCloudSize, calibration)
            calibrator.out.link(self._calibration_input)
        else:
            if selectCandidate is not None:
                raise TypeError(
                    "selectCandidate requires a legacy PlaneCalibrationConfig"
                )
            calibrator = self.createSubnode(SupportPlaneCalibration)
            calibrator.build(pointCloud, imu, pointCloudSize, calibration)
            calibrator.candidatesOutput.link(self._calibration_input)
            self._request_output.link(calibrator.inputVerification)
        self._control_output.link(calibrator.inputControl)
        calibrator.inputControl.setMaxSize(CONTROL_QUEUE_SIZE)
        calibrator.verificationOutput.link(self._verification_input)
        self._calibrator = calibrator
        if selectCandidate is not None:
            self._select_candidate = selectCandidate
        self._plane = initialPlane
        self._config = config
        return self

    def run(self) -> None:
        """Drive the calibrator and send results until stopped."""
        try:
            while self.isRunning():
                self._run_pass()
                # Pause between passes, as DepthAI's DynamicCalibration does.
                time.sleep(0.005)
        # DepthAI's stub declares QueueException through an undeclared nested
        # class, so pyright cannot tell it is an exception; at runtime it is one.
        except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
            return

    def _run_pass(self) -> None:
        """Start once, handle the calibrator's answers, then check."""
        if not self._started:
            self._started = True
            if self._plane is None:
                self._start_search()
            else:
                self._gate_pending = True
                self._send_verify(self._plane)
        while (result := self._calibration_input.tryGet()) is not None:
            if isinstance(result, SupportPlaneCandidateBatch):
                self._handle_batch(result)
            elif isinstance(result, SupportPlaneCalibrationResult):
                self._handle_calibration(result)
        while (verification := self._verification_input.tryGet()) is not None:
            assert isinstance(verification, SupportPlaneVerificationResult)
            self._handle_verification(verification)
        self._check_plane_if_due()

    def _start_search(self) -> None:
        """Start a calibration attempt, keeping any current plane."""
        self._searching = True
        self._control_output.send(SupportPlaneCalibrationControl.start())

    def _handle_batch(self, batch: SupportPlaneCandidateBatch) -> None:
        passing = [
            assessment for assessment in batch.assessments if assessment.accepted
        ]
        if not self._searching or not passing:
            return
        selected = max(
            passing,
            key=lambda item: (
                item.on_plane_point_count,
                -item.perpendicular_distance_mm,
            ),
        )
        self._searching = False
        self._hold(selected.plane)
        self._send(AutoSupportPlaneResult(True, selected.plane), batch)

    def _handle_calibration(self, result: SupportPlaneCalibrationResult) -> None:
        """Reset the calibrator and report the attempt's plane, or its failure.

        A selector that returns ``None`` starts a new attempt and sends nothing.
        """
        self._control_output.send(SupportPlaneCalibrationControl.reset())
        self._searching = False
        if result.outcome is CalibrationOutcome.SUCCEEDED:
            selected = self._select_candidate(result)
            if selected is None:
                self._start_search()
                return
            if selected not in result.candidates:
                raise ValueError(
                    "selectCandidate must return one of the candidates or None, "
                    f"got {selected!r}"
                )
            self._hold(selected.plane)
            self._send(AutoSupportPlaneResult(True, selected.plane), result)
        elif self._plane is not None:
            self._hold(self._plane)
            self._send(AutoSupportPlaneResult(False, self._plane), result)
        else:
            self._send(AutoSupportPlaneResult(False, None), result)
            if self._continuous:
                self._start_search()

    def _hold(self, plane: SupportPlane) -> None:
        """Make ``plane`` current, with its first plane check one interval away."""
        self._plane = plane
        self._failed_checks = 0
        self._schedule_check()

    def _schedule_check(self) -> None:
        """Make the next plane check due one interval from now."""
        assert self._config is not None
        self._next_check_at = time.monotonic() + self._config.check_interval_sec

    def _handle_verification(self, result: SupportPlaneVerificationResult) -> None:
        """Judge the ``initialPlane`` gate or a plane check.

        A rejected gate drops the plane and searches; failed checks refit after enough
        failures in a row.

        Answers that arrive during a calibration attempt, or for a plane other than the
        held one, are stale and ignored.
        """
        assert self._config is not None
        if isinstance(self._calibrator, SupportPlaneCalibration) and (
            not self._verify_in_flight or result.request_id != self._request_id
        ):
            return
        self._verify_in_flight = False
        gate, self._gate_pending = self._gate_pending, False
        if self._searching or result.plane != self._plane:
            return
        self._schedule_check()
        if result.accepted:
            self._failed_checks = 0
            self._send(AutoSupportPlaneResult(True, result.plane), result)
            return
        if gate:
            self._plane = None
            self._send(AutoSupportPlaneResult(False, None), result)
            self._start_search()
            return
        self._failed_checks += 1
        if self._failed_checks >= self._config.failed_checks_before_refit:
            self._failed_checks = 0
            self._start_search()

    def _check_plane_if_due(self) -> None:
        """Send ``Verify(current plane)`` in ``CONTINUOUS`` once the interval passed."""
        if (
            not self._continuous
            or self._plane is None
            or self._searching
            or self._verify_in_flight
            or time.monotonic() < self._next_check_at
        ):
            return
        self._send_verify(self._plane)

    def _send_verify(self, plane: SupportPlane) -> None:
        self._verify_in_flight = True
        if isinstance(self._calibrator, SupportPlaneCalibration):
            self._request_id += 1
            self._request_output.send(
                SupportPlaneVerificationRequest(plane, self._request_id)
            )
        else:
            self._control_output.send(SupportPlaneCalibrationControl.verify(plane))

    @property
    def _continuous(self) -> bool:
        assert self._config is not None
        return self._config.mode is AutoSupportPlaneConfig.Mode.CONTINUOUS

    def _send(self, message: AutoSupportPlaneResult, trigger: dai.Buffer) -> None:
        """Send a result stamped from the calibrator result that triggered it."""
        message.setBufferMetadataFrom(trigger)
        self._out.send(message)


def _select_largest_plane(
    result: SupportPlaneCalibrationResult,
) -> SupportPlaneCandidate:
    """Return the candidate with the most inliers, ties to the nearest plane."""
    return max(
        result.candidates,
        key=lambda c: (c.inlier_count, -abs(c.plane.offset_mm)),
    )

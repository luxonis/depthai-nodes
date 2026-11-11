import time
from collections.abc import Callable, Sequence

import depthai as dai

from depthai_nodes.message.auto_support_plane_result import AutoSupportPlaneResult
from depthai_nodes.message.support_plane import SupportPlane, SupportPlaneCandidate
from depthai_nodes.message.support_plane_calibration import (
    CalibrationOutcome,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
    SupportPlaneVerificationResult,
)
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode

from .auto_support_plane_config import AutoSupportPlaneConfig
from .support_plane_calibrator import PASS_SLEEP_SECONDS, SupportPlaneCalibrator

CandidateSelector = Callable[
    [tuple[SupportPlaneCandidate, ...]], SupportPlaneCandidate | None
]

# The calibrator's control input size. The wrapper has at most a Reset, a Start
# and one Verify pending there at once, so 8 leaves headroom over those 3 and
# its sends never block.
CONTROL_QUEUE_SIZE = 8


class AutoSupportPlane(BaseThreadedHostNode):
    """Find and hold the support plane from a point cloud and IMU.

    Wraps a `SupportPlaneCalibrator`. ``ON_START`` finds a plane once and holds
    it; ``CONTINUOUS`` also checks it about every ``check_interval_sec`` and
    refits after repeated failed checks, keeping the old plane meanwhile. Pass
    the plane held before a pipeline rebuild as ``initialPlane``.

    Attributes:
        out: Output stream of `depthai_nodes.message.AutoSupportPlaneResult`,
            one per found, checked or rejected plane.
    """

    def __init__(self) -> None:
        """Create the inner calibrator, its internal links and the output."""
        super().__init__()
        self._calibrator: SupportPlaneCalibrator = self.createSubnode(
            SupportPlaneCalibrator
        )
        self._control_output = self.createOutput(name="calibratorControl")
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
            initialPlane: Held at once, then verified on the first cloud with
                fresh IMU; dropped for a search if rejected.
            selectCandidate: Picks the plane from a successful attempt's
                candidates, or returns ``None`` to keep searching; runs on the
                node's thread. ``None`` picks the most inliers.

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
            config = AutoSupportPlaneConfig()
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
        self._calibrator.build(pointCloud, imu, pointCloudSize, config.calibration)
        self._control_output.link(self._calibrator.inputControl)
        self._calibrator.inputControl.setMaxSize(CONTROL_QUEUE_SIZE)
        self._calibrator.out.link(self._calibration_input)
        self._calibrator.verificationOutput.link(self._verification_input)
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
                time.sleep(PASS_SLEEP_SECONDS)
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
            assert isinstance(result, SupportPlaneCalibrationResult)
            self._handle_calibration(result)
        while (verification := self._verification_input.tryGet()) is not None:
            assert isinstance(verification, SupportPlaneVerificationResult)
            self._handle_verification(verification)
        self._check_plane_if_due()

    def _start_search(self) -> None:
        """Start a calibration attempt, keeping any current plane."""
        self._searching = True
        self._control_output.send(SupportPlaneCalibrationControl.start())

    def _handle_calibration(self, result: SupportPlaneCalibrationResult) -> None:
        """Reset the calibrator and report the attempt's plane, or its failure.

        A selector that returns ``None`` starts a new attempt and sends nothing.
        """
        self._control_output.send(SupportPlaneCalibrationControl.reset())
        self._searching = False
        if result.outcome is CalibrationOutcome.SUCCEEDED:
            selected = self._select_candidate(result.candidates)
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

        A rejected gate drops the plane and searches; failed checks refit
        after enough failures in a row.

        Answers outside ``VALID`` or for another plane are stale and ignored.
        """
        assert self._config is not None
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
        self._control_output.send(SupportPlaneCalibrationControl.verify(plane))

    @property
    def _continuous(self) -> bool:
        assert self._config is not None
        return self._config.mode is AutoSupportPlaneConfig.Mode.CONTINUOUS

    def _send(self, message: AutoSupportPlaneResult, trigger: dai.Buffer) -> None:
        """Send a result stamped from the calibrator result that triggered it."""
        message.setTimestamp(trigger.getTimestamp())
        message.setTimestampDevice(trigger.getTimestampDevice())
        message.setSequenceNum(trigger.getSequenceNum())
        self._out.send(message)


def _select_largest_plane(
    candidates: Sequence[SupportPlaneCandidate],
) -> SupportPlaneCandidate:
    """Return the candidate with the most inliers, ties to the nearest plane."""
    return max(candidates, key=lambda c: (c.inlier_count, -abs(c.plane.offset_mm)))

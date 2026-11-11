import time
from collections.abc import Sequence

import depthai as dai

from depthai_nodes.message.auto_support_plane_result import AutoSupportPlaneResult
from depthai_nodes.message.support_plane import SupportPlane, SupportPlaneCandidate
from depthai_nodes.message.support_plane_calibration import (
    CalibrationOutcome,
    SupportPlaneCalibrationControl,
    SupportPlaneCalibrationResult,
)
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.support_plane_calibrator.support_plane_calibrator import (
    PASS_SLEEP_SECONDS,
    SupportPlaneCalibrator,
    _check_instance,
    _copy_stamps,
)

from .auto_support_plane_config import AutoSupportPlaneConfig

# Room on the calibrator's control input for a Reset, a Start and a Verify
# pending together, so the wrapper's sends never block.
CONTROL_QUEUE_SIZE = 8


class AutoSupportPlane(BaseThreadedHostNode):
    """Find and hold the support plane, driving an inner ``SupportPlaneCalibrator``.

    Link a point cloud and an IMU at `build` and read
    `depthai_nodes.message.AutoSupportPlaneResult` messages from ``out``. In
    ``ON_START`` mode the node runs one calibration attempt and holds the plane
    it selects, sending ``(True, plane)``. If the attempt ends at
    ``calibration.max_calibration_attempts`` without a plane, it sends
    ``(False, None)`` and does nothing more until the pipeline is rebuilt.
    Nothing is sent while the attempt runs, so without an attempt cap the node
    searches until a plane comes into view.

    Of a successful attempt's support plane candidates, the node selects the
    one with the most ``inlier_count`` (roughly the largest visible plane),
    ties going to the smallest ``|offset_mm|`` (the plane nearest the camera).

    Drive `SupportPlaneCalibrator` directly instead for a person picking among
    candidates or any custom flow. Requires open3d, installed with
    ``pip install "depthai-nodes[box-measurement]"``. Nothing logs.
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
        self._started = False

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
    ) -> "AutoSupportPlane":
        """Build the inner calibrator on the point cloud and IMU.

        Args:
            pointCloud: Output emitting organized ``dai.PointCloudData`` in
                millimeters.
            imu: Output emitting ``dai.IMUData`` with accelerometer packets.
            pointCloudSize: Cloud ``(width, height)`` in points.
            config: Mode and calibrator settings; ``None`` uses the defaults.

        Returns:
            The configured node instance.

        Raises:
            ImportError: If open3d is not installed.
            RuntimeError: If the node is already built.
            TypeError: If ``config`` is not an ``AutoSupportPlaneConfig``, or
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
        _check_instance("config", config, AutoSupportPlaneConfig)
        self._calibrator.build(pointCloud, imu, pointCloudSize, config.calibration)
        self._control_output.link(self._calibrator.inputControl)
        self._calibrator.inputControl.setMaxSize(CONTROL_QUEUE_SIZE)
        self._calibrator.out.link(self._calibration_input)
        self._calibrator.verificationOutput.link(self._verification_input)
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
        """Start the search on the first pass, then handle calibration results."""
        if not self._started:
            self._started = True
            self._control_output.send(SupportPlaneCalibrationControl.start())
        while True:
            result = self._calibration_input.tryGet()
            if result is None:
                return
            assert isinstance(result, SupportPlaneCalibrationResult)
            self._handle_calibration(result)

    def _handle_calibration(self, result: SupportPlaneCalibrationResult) -> None:
        """Reset the calibrator and report the attempt's plane, or its failure."""
        self._control_output.send(SupportPlaneCalibrationControl.reset())
        plane: SupportPlane | None = None
        if result.outcome is CalibrationOutcome.SUCCEEDED:
            plane = _select_largest_plane(result.candidates).plane
        self._send(AutoSupportPlaneResult(plane is not None, plane), result)

    def _send(self, message: AutoSupportPlaneResult, trigger: dai.Buffer) -> None:
        """Send a result stamped from the calibrator result that triggered it."""
        _copy_stamps(trigger, message)
        self._out.send(message)


def _select_largest_plane(
    candidates: Sequence[SupportPlaneCandidate],
) -> SupportPlaneCandidate:
    """Return the candidate with the most inliers, ties to the nearest plane."""
    return max(candidates, key=lambda c: (c.inlier_count, -abs(c.plane.offset_mm)))

import math
from datetime import timedelta

import depthai as dai
import pytest

from depthai_nodes.message import (
    CameraMovementCalibrationState,
    CameraMovementReason,
)
from depthai_nodes.node import CameraMovementConfig, CameraMovementDetector
from tests.utils import InputMock, OutputMock, PipelineMock

GRAVITY = 9.80665
STILL = (0, 0, 0)
LEVEL = (0, 0, GRAVITY)


def set_report(report, ms, vector):
    timestamp = dai.Timestamp()
    timestamp.sec, milliseconds = divmod(ms, 1000)
    timestamp.nsec = milliseconds * 1_000_000
    report.tsDevice = timestamp
    report.timestamp = timestamp
    report.x, report.y, report.z = vector


class Detector(CameraMovementDetector):
    """Drive the detector with deterministic report timestamps."""

    def __init__(self):
        super().__init__()
        self._config = CameraMovementConfig(
            turn_threshold_deg=15, calibration_window_s=0.5
        )
        self.events = []

    def request_calibration(self, ms):
        self._calibration = CameraMovementCalibrationState.COLLECTING
        self._calibration_after = timedelta(milliseconds=ms)
        self._calibration_start = None

    def sample(self, ms, accel=LEVEL, gyro=STILL):
        packet = dai.IMUPacket()
        if gyro is not None:
            set_report(packet.gyroscope, ms, gyro)
        if accel is not None:
            set_report(packet.acceleroMeter, ms, accel)
        self._process_packet(packet, ms / 1000)

    def _send_event(self, reason, angle, report):
        self.events.append((reason, angle, report.getTimestampDevice()))

    def calibrate(self):
        self.request_calibration(1000)
        for ms in range(1000, 1610, 10):
            self.sample(ms)
        assert self._calibration is CameraMovementCalibrationState.SUCCEEDED


@pytest.fixture
def detector():
    # The threaded-node suite can replace the SDK classes at collection time.
    # Use a native host-only pipeline when available and mock ports otherwise.
    if dai.Pipeline is PipelineMock:

        class MockDetector(Detector):
            def createInput(self, *args, **kwargs):
                return InputMock()

            def createOutput(self, *args, **kwargs):
                return OutputMock()

        yield MockDetector()
    else:
        with dai.Pipeline(False) as pipeline:
            yield pipeline.create(Detector)


def test_compound_tilt_preserves_turn_projection(detector):
    detector.calibrate()
    rate = math.pi / 2

    # Rotate 90 degrees around the local x axis. Gravity goes from local
    # +z to +y. The local z axis is now horizontal and can be turned about.
    for step in range(1, 101):
        angle = rate * step / 100
        accel = (0, GRAVITY * math.sin(angle), GRAVITY * math.cos(angle))
        detector.sample(1600 + step * 10, accel, (rate, 0, 0))
    for step in range(1, 101):
        angle = rate * step / 100
        accel = (GRAVITY * math.sin(angle), GRAVITY * math.cos(angle), 0)
        detector.sample(2600 + step * 10, accel, (0, 0, rate))
    for ms in range(3610, 4710, 10):
        detector.sample(ms, (GRAVITY, 0, 0))
    assert [reason for reason, _, _ in detector.events] == [
        CameraMovementReason.TILT_CHANGED
    ]

    for ms in range(4710, 5110, 10):
        detector.sample(ms, (GRAVITY, 0, 0), (rate, 0, 0))
    for ms in range(5110, 5710, 10):
        detector.sample(ms, (GRAVITY, 0, 0))
    assert [reason for reason, _, _ in detector.events] == [
        CameraMovementReason.TILT_CHANGED,
        CameraMovementReason.TURN_DETECTED,
    ]
    assert detector.events[-1][1] == pytest.approx(36, abs=0.1)


@pytest.mark.parametrize("missing_sensor", ["accel", "gyro"])
def test_gap_restarts_calibration_window(detector, missing_sensor):
    detector.request_calibration(1000)
    for ms in range(1000, 1920, 10):
        dropped = {missing_sensor: None} if 1460 <= ms < 1560 else {}
        detector.sample(ms, **dropped)
    assert detector._calibration is CameraMovementCalibrationState.COLLECTING
    for ms in range(1920, 2110, 10):
        detector.sample(ms)
    assert detector._calibration is CameraMovementCalibrationState.SUCCEEDED


@pytest.mark.parametrize("missing_sensor", ["accel", "gyro"])
def test_gap_restarts_tilt_confirmation(detector, missing_sensor):
    detector.calibrate()
    for ms in range(1610, 2290, 10):
        reports = {"accel": (GRAVITY, 0, 0), "gyro": STILL}
        if 2110 <= ms < 2210:
            reports[missing_sensor] = None
        detector.sample(ms, **reports)
    assert detector.events == []
    for ms in range(2290, 2810, 10):
        detector.sample(ms, (GRAVITY, 0, 0))
    assert len(detector.events) == 1
    assert detector.events[0][2] >= timedelta(milliseconds=2710)


def test_yaw_above_stationary_gate_cannot_become_calibration_bias(detector):
    detector.request_calibration(1000)
    for ms in range(1000, 1710, 10):
        detector.sample(ms, gyro=(0, 0, math.radians(4)))
    assert detector._calibration is CameraMovementCalibrationState.COLLECTING

    for ms in range(1710, 2310, 10):
        detector.sample(ms)
    assert detector._calibration is CameraMovementCalibrationState.SUCCEEDED
    for ms in range(2310, 2710, 10):
        detector.sample(ms, gyro=(0, 0, math.radians(50)))
    for ms in range(2710, 3310, 10):
        detector.sample(ms)
    assert len(detector.events) == 1
    assert detector.events[0][0] is CameraMovementReason.TURN_DETECTED
    assert detector.events[0][1] == pytest.approx(20)


@pytest.mark.parametrize(
    ("degrees_per_second", "duration_ms"), [(20 / 12, 12000), (4, 15000)]
)
def test_turn_must_meet_rate_and_duration_limits(
    detector, degrees_per_second, duration_ms
):
    detector.calibrate()
    start = 1610
    end = start + duration_ms
    for ms in range(start, end, 10):
        detector.sample(ms, gyro=(0, 0, math.radians(degrees_per_second)))
    for ms in range(end, end + 600, 10):
        detector.sample(ms)
    assert detector.events == []

import math
import subprocess
import sys
from collections import deque
from datetime import datetime, timedelta
from types import SimpleNamespace

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import (
    SupportPlane,
    SupportPlaneCalibrationControl,
    SupportPlaneCandidateBatch,
    SupportPlaneVerificationRequest,
    SupportPlaneVerificationResult,
)
from depthai_nodes.node import SupportPlaneCalibration, SupportPlaneCalibrationConfig
from depthai_nodes.node.support_plane_calibrator import support_plane_calibration
from tests.utils.scripted_open3d import ScriptedScene

CLOUD_SIZE = (11, 9)
BELOW = SupportPlane(normal=(0, 0, 1), offset_mm=-800)
HOST_STAMP = timedelta(seconds=12, microseconds=34)
DEVICE_STAMP = timedelta(seconds=3, microseconds=56)
IMU_STAMP = timedelta(seconds=9, microseconds=50_000)
IMU_HOST_STAMP = timedelta(seconds=29, microseconds=60_000)
SYSTEM_STAMP = datetime(2025, 5, 6, 12, 15, 28)
IMU_SYSTEM_STAMP = datetime.fromtimestamp(1746533727)


def cloud_message(sequence, size=CLOUD_SIZE, depths=800, *, system_stamp=None):
    width, height = size
    x, y = np.meshgrid(np.arange(width) * 40, np.arange(height) * 40)
    z = np.broadcast_to(depths, (height, width))
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3).astype(np.float32)
    cloud = dai.PointCloudData().setSize(width, height)
    cloud.setPoints(points)
    cloud.setSequenceNum(sequence)
    cloud.setTimestamp(HOST_STAMP)
    cloud.setTimestampDevice(DEVICE_STAMP)
    cloud.setTimestampSystem(system_stamp)
    return cloud


def imu_message(*accelerations):
    packets = []
    for index, z in enumerate(accelerations):
        packet = dai.IMUPacket()
        packet.acceleroMeter.z = z
        stamp = dai.Timestamp()
        stamp.sec = 8 + index
        stamp.nsec = 50_000_000
        packet.acceleroMeter.tsDevice = stamp
        synced_stamp = dai.Timestamp()
        synced_stamp.sec = 28 + index
        synced_stamp.nsec = 60_000_000
        packet.acceleroMeter.timestamp = synced_stamp
        if hasattr(packet.acceleroMeter, "tsSystem"):
            system_stamp = dai.Timestamp()
            system_stamp.sec = 1746533726 + index
            packet.acceleroMeter.tsSystem = system_stamp
        packets.append(packet)
    imu = dai.IMUData()
    imu.packets = packets
    imu.setTimestampDevice(timedelta(seconds=21))
    return imu


class Queue:
    def __init__(self, max_size=None):
        self.messages = deque(maxlen=max_size)
        self.callback = None

    def send(self, message):
        if self.callback is not None:
            self.callback(message)
        self.messages.append(message)

    def addCallback(self, callback):
        self.callback = callback

    def tryGet(self):
        return self.messages.popleft() if self.messages else None

    def tryGetAll(self):
        messages = list(self.messages)
        self.messages.clear()
        return messages


class Source:
    def __init__(self, node_id, name):
        self.parent = SimpleNamespace(id=node_id)
        self.name = name

    def getParent(self):
        return self.parent

    def getGroup(self):
        return "synthetic"

    def getName(self):
        return self.name

    def link(self, target):
        self.target = target

    def send(self, message):
        self.target.send(message)


class ManualCalibration(SupportPlaneCalibration):
    def createInput(self, *args, blocking=True, queueSize=3, **kwargs):
        return Queue(max_size=None if blocking else queueSize)

    def createOutput(self, *args, **kwargs):
        return Queue()


class Harness:
    def __init__(self, node, monkeypatch):
        self.cloud = Source(1, "cloud")
        self.imu = Source(2, "imu")
        self.node = node
        self.time = 10.0
        self.monkeypatch = monkeypatch
        monkeypatch.setattr(
            support_plane_calibration,
            "time",
            SimpleNamespace(monotonic=lambda: self.time),
        )

    def build(self, **config):
        self.node.build(
            self.cloud, self.imu, CLOUD_SIZE, SupportPlaneCalibrationConfig(**config)
        )
        return self

    def step(self, elapsed=0):
        self.time += elapsed
        self.node._step()

    def fit(self, planes, *, during_fit=None):
        self.monkeypatch.setattr(
            "open3d.geometry.PointCloud",
            ScriptedScene(planes, during_fit=during_fit).cloud,
        )

    def start(self):
        self.node.inputControl.send(SupportPlaneCalibrationControl.start())

    def stop(self):
        self.node.inputControl.send(SupportPlaneCalibrationControl.stop())

    def verify(self, plane, request_id=None):
        self.node.inputVerification.send(
            SupportPlaneVerificationRequest(plane, request_id)
        )

    def batch(self):
        return self.node.candidatesOutput.tryGet()

    def reply(self):
        return self.node.verificationOutput.tryGet()


def fit_interval_latest_cloud_and_restart(manual):
    h = manual.build()
    h.start()
    first_cloud = cloud_message(1)
    h.cloud.send(first_cloud)
    h.step()
    assert h.batch() is None

    def elapsed_fit():
        h.time += 0.125

    h.fit([((0, 0, 1, -800), 0)], during_fit=elapsed_fit)
    h.imu.send(imu_message(-9.8))
    h.step()
    first = h.batch()
    assert isinstance(first, SupportPlaneCandidateBatch)
    assert first.getSequenceNum() == 1
    assert first.assessments == ()

    h.fit([((0, 0, 1, -800), 80)])
    h.start()
    h.cloud.send(cloud_message(2))
    h.cloud.send(cloud_message(3))
    h.step(0.25)
    assert h.batch() is None

    h.step(0.125)
    second = h.batch()
    assert second.getSequenceNum() == 3
    assert len(second.assessments) == 1
    assert not second.assessments[0].accepted
    assert second.assessments[0].reasons[0].code == "insufficient_sample"

    h.imu.send(imu_message(-9.8))
    h.step(0.5)
    assert h.batch() is None

    h.cloud.send(first_cloud)
    h.step()
    assert h.batch() is None

    h.stop()
    h.step()
    assert h.batch() is None
    h.start()
    h.step()
    assert h.batch().getSequenceNum() == 1


def stop_during_fit_allows_that_batch_and_blocks_next_fit(manual):
    h = manual.build()
    h.imu.send(imu_message(-9.8))
    h.cloud.send(cloud_message(1))
    h.start()
    h.fit([((0, 0, 1, -800), 80)], during_fit=h.stop)
    h.step()
    assert h.batch().getSequenceNum() == 1

    h.fit([((0, 0, 1, -800), 80)])
    h.imu.send(imu_message(-9.8))
    h.cloud.send(cloud_message(2))
    h.step(0.5)
    assert h.batch() is None

    h.start()
    h.step()
    assert h.batch().getSequenceNum() == 2


def gravity_expires_independently_of_the_fit_interval(manual):
    h = manual.build(attempt_interval_sec=1)
    h.fit([((0, 0, 1, -800), 80)])
    h.cloud.send(cloud_message(1))
    h.imu.send(imu_message(-9.8))
    h.start()
    h.step()
    assert h.batch().getSequenceNum() == 1

    h.cloud.send(cloud_message(2))
    h.imu.send(imu_message(0))
    h.verify(BELOW, "waiting")
    h.step(1)
    assert h.batch() is None
    assert h.reply() is None

    h.cloud.send(cloud_message(3))
    h.imu.send(imu_message(-9.8))
    h.imu.send(imu_message(0))
    h.step()
    assert h.batch().getSequenceNum() == 3
    reply = h.reply()
    assert reply.request_id == "waiting"
    assert reply.getSequenceNum() == 3


def full_mixed_batch_and_verification_share_snapshot_assessment(manual):
    h = manual.build(min_sampled_point_count=8, min_on_plane_ratio=0.5)
    h.fit(
        [
            ((0, 0, 1, -1200), 30),
            ((0, 0, 1, -800), 30),
            ((0, 0, -2, 1600), 20),
            ((1, 0, 0, -400), 15),
        ]
    )
    depths = np.full((CLOUD_SIZE[1], CLOUD_SIZE[0]), 1200.0)
    depths[::4, ::4] = [[800, 830, 800], [0, 800, 800], [900, 800, 900]]
    cloud = cloud_message(5, depths=depths, system_stamp=SYSTEM_STAMP)
    h.cloud.send(cloud)
    h.imu.send(imu_message(9.8, -9.8, 0))
    h.start()
    h.step()

    batch = h.batch()
    assert isinstance(batch, SupportPlaneCandidateBatch)
    assert len(batch.assessments) == 3
    assessments = {
        result.perpendicular_distance_mm: result for result in batch.assessments
    }
    accepted, rejected, tilted = (
        assessments[distance] for distance in (800, 1200, 400)
    )
    assert accepted.accepted
    assert accepted.reasons == ()
    assert accepted.sampled_point_count == 8
    assert accepted.on_plane_point_count == 6
    assert accepted.on_plane_ratio == 0.75
    assert accepted.tilt_deg == 0
    assert accepted.perpendicular_distance_mm == 800
    assert accepted.vertical_distance_mm == 800
    (reason,) = rejected.reasons
    assert (reason.code, reason.measured, reason.limit) == (
        "coverage_below_minimum",
        0,
        0.5,
    )
    assert "tilt_above_maximum" in {reason.code for reason in tilted.reasons}

    h.verify(rejected.plane, "rejected")
    h.verify(accepted.plane, "accepted")
    h.verify(tilted.plane, "tilted")
    h.step()
    for expected, request_id in (
        (rejected, "rejected"),
        (accepted, "accepted"),
        (tilted, "tilted"),
    ):
        reply = h.reply()
        assert isinstance(reply, SupportPlaneVerificationResult)
        assert reply.request_id == request_id
        assert reply.assessment == expected
        assert reply.source_stream_id == batch.source_stream_id
        assert reply.gravity_direction == batch.gravity_direction
        assert reply.imu_timestamp_device == batch.imu_timestamp_device
        assert reply.imu_timestamp_host == batch.imu_timestamp_host
        assert reply.imu_timestamp_system == batch.imu_timestamp_system
        assert reply.getSequenceNum() == batch.getSequenceNum() == 5
        assert reply.getTimestamp() == batch.getTimestamp() == HOST_STAMP
        assert reply.getTimestampDevice() == batch.getTimestampDevice() == DEVICE_STAMP
        assert reply.getTimestampSystem() == batch.getTimestampSystem() == SYSTEM_STAMP

    assert batch.source_stream_id == "1/synthetic/cloud"
    assert batch.gravity_direction == pytest.approx((0, 0, 1))
    assert batch.imu_timestamp_device == IMU_STAMP
    assert batch.imu_timestamp_host == IMU_HOST_STAMP
    expected_imu_system = (
        IMU_SYSTEM_STAMP if hasattr(dai.IMUReportAccelerometer, "tsSystem") else None
    )
    assert batch.imu_timestamp_system == expected_imu_system
    assert h.reply() is None

    h.cloud.send(cloud_message(6))
    h.imu.send(imu_message(-9.8))
    h.step(0.5)
    assert h.batch() is None
    h.start()
    h.step()
    restarted = h.batch()
    assert restarted.getSequenceNum() == 6
    assert restarted.getTimestampSystem() is None


def verification_waits_replies_in_order_and_does_not_control_search(manual):
    h = manual.build(min_sampled_point_count=1)
    h.fit([((0, 0, 1, -1200), 80)])
    h.verify(BELOW)
    h.verify(BELOW, "second")
    h.imu.send(imu_message(-9.8))
    h.step()
    assert h.reply() is None
    assert h.batch() is None

    h.cloud.send(cloud_message(10, size=CLOUD_SIZE[::-1]))
    h.step()
    assert h.reply() is None
    assert h.batch() is None

    h.cloud.send(cloud_message(1))
    h.step()
    for request_id in (None, "second"):
        reply = h.reply()
        assert reply.request_id == request_id
        assert reply.assessment.accepted
    assert h.reply() is None
    assert h.batch() is None

    h.start()
    h.step()
    assert not h.batch().assessments[0].accepted

    h.cloud.send(cloud_message(2))
    h.verify(BELOW, "while-searching")
    h.step(0.25)
    reply = h.reply()
    assert reply.request_id == "while-searching"
    assert reply.assessment.accepted
    assert reply.getSequenceNum() == 2
    assert h.batch() is None
    assert h.reply() is None

    h.step(0.25)
    assert h.batch().getSequenceNum() == 2


def insufficient_sample_still_reports_counts_and_geometry_failures(manual):
    h = manual.build(min_sampled_point_count=2, vertical_distance_range_mm=(700, 900))
    depths = np.full((CLOUD_SIZE[1], CLOUD_SIZE[0]), 0.0)
    depths[0, 0] = 1000
    h.cloud.send(cloud_message(1, depths=depths))
    h.imu.send(imu_message(-9.8))
    h.verify(SupportPlane(normal=(0, 0, 1), offset_mm=-1000))
    h.step()
    result = h.reply().assessment
    assert (result.sampled_point_count, result.on_plane_point_count) == (1, 1)
    assert result.on_plane_ratio == 1
    assert result.vertical_distance_mm == 1000
    assert not result.accepted
    reasons = {reason.code: reason for reason in result.reasons}
    assert set(reasons) == {"insufficient_sample", "vertical_distance_out_of_range"}
    assert reasons["insufficient_sample"].measured == 1
    assert reasons["insufficient_sample"].limit == 2
    assert reasons["vertical_distance_out_of_range"].measured == 1000
    assert reasons["vertical_distance_out_of_range"].limit == (700, 900)

    h.cloud.send(cloud_message(2, depths=0))
    h.verify(SupportPlane(normal=(1, 0, 0), offset_mm=-100))
    h.step()
    result = h.reply().assessment
    assert (result.sampled_point_count, result.on_plane_point_count) == (0, 0)
    assert result.on_plane_ratio is None
    assert result.tilt_deg == 90
    assert result.vertical_distance_mm is None
    assert {
        "insufficient_sample",
        "tilt_above_maximum",
        "vertical_distance_unavailable",
    } <= {reason.code for reason in result.reasons}


def vertical_distance_range_is_inclusive_and_along_gravity(manual):
    h = manual.build(
        max_gravity_angle_deg=45,
        min_sampled_point_count=1,
        min_on_plane_ratio=0,
        vertical_distance_range_mm=(700, 900),
    )
    h.cloud.send(cloud_message(1))
    h.imu.send(imu_message(-9.8))
    h.verify(SupportPlane(normal=(0, 1, 2), offset_mm=-1400))
    h.verify(SupportPlane(normal=(0, -1, -2), offset_mm=1800))
    angle = math.radians(6.5)
    normal = (0, math.sin(angle), math.cos(angle))
    h.verify(SupportPlane(normal=normal, offset_mm=-700 * normal[2]))
    h.verify(SupportPlane(normal=normal, offset_mm=-699.99 * normal[2]))
    h.verify(SupportPlane(normal=(0, 0, 1), offset_mm=800))
    h.step()

    for distance in (700, 900):
        assessment = h.reply().assessment
        assert assessment.vertical_distance_mm == distance
        assert assessment.perpendicular_distance_mm == pytest.approx(
            distance * 2 / np.sqrt(5)
        )
        assert assessment.accepted

    boundary = h.reply().assessment
    assert boundary.vertical_distance_mm == pytest.approx(700)
    assert boundary.accepted
    out_of_range = h.reply().assessment
    assert not out_of_range.accepted
    assert out_of_range.reasons[0].code == "vertical_distance_out_of_range"

    above = h.reply().assessment
    assert above.vertical_distance_mm is None
    assert not above.accepted
    assert "plane_not_below_camera" in {reason.code for reason in above.reasons}


@pytest.mark.parametrize(
    "case",
    [
        fit_interval_latest_cloud_and_restart,
        stop_during_fit_allows_that_batch_and_blocks_next_fit,
        gravity_expires_independently_of_the_fit_interval,
        full_mixed_batch_and_verification_share_snapshot_assessment,
        verification_waits_replies_in_order_and_does_not_control_search,
        insufficient_sample_still_reports_counts_and_geometry_failures,
        vertical_distance_range_is_inclusive_and_along_gravity,
    ],
    ids=lambda case: case.__name__,
)
def test_controlled_calibration(case, monkeypatch, request):
    pytest.importorskip("open3d")
    # Legacy node tests replace the SDK's Pipeline globally during collection.
    if dai.Pipeline.__module__ != "depthai":
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", request.node.nodeid],
            check=True,
            timeout=30,
        )
        return
    with dai.Pipeline(False) as pipeline:
        case(Harness(pipeline.create(ManualCalibration), monkeypatch))


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_fit_interval_requires_positive_finite_seconds(interval):
    with pytest.raises(ValueError):
        SupportPlaneCalibrationConfig(attempt_interval_sec=interval)


def test_native_pipeline_fits_and_verifies_a_real_planar_cloud():
    pytest.importorskip("open3d")
    subprocess.run([sys.executable, "-m", __name__], check=True, timeout=30)


def native_pipeline_check():
    class SensorSource(dai.node.ThreadedHostNode):
        def __init__(self):
            super().__init__()
            self.cloud = self.createOutput("cloud")
            self.imu = self.createOutput("imu")

        def run(self):
            pass

    size = (96, 96)
    with dai.Pipeline(False) as pipeline:
        source = pipeline.create(SensorSource)
        node = pipeline.create(SupportPlaneCalibration).build(
            source.cloud, source.imu, size
        )
        control = node.inputControl.createInputQueue()
        requests = node.inputVerification.createInputQueue()
        batches = node.candidatesOutput.createOutputQueue()
        checks = node.verificationOutput.createOutputQueue()
        pipeline.start()
        source.cloud.send(cloud_message(101, size=size, system_stamp=SYSTEM_STAMP))
        source.imu.send(imu_message(-9.8, -9.8))
        source.imu.send(imu_message(0))
        control.send(SupportPlaneCalibrationControl.start())

        batch = batches.get(timedelta(seconds=5))
        assert isinstance(batch, SupportPlaneCandidateBatch)
        accepted = next(result for result in batch.assessments if result.accepted)
        assert accepted.sampled_point_count == accepted.on_plane_point_count == 576
        assert accepted.vertical_distance_mm == pytest.approx(800)
        assert batch.getTimestampSystem() == SYSTEM_STAMP

        requests.send(SupportPlaneVerificationRequest(accepted.plane, "native"))
        source.imu.send(imu_message(-9.8, -9.8))
        reply = checks.get(timedelta(seconds=5))
        assert isinstance(reply, SupportPlaneVerificationResult)
        assert reply.assessment == accepted
        assert reply.request_id == "native"
        assert reply.getSequenceNum() == 101
        assert reply.getTimestampSystem() == SYSTEM_STAMP
        assert reply.imu_timestamp_device == IMU_STAMP
        assert reply.imu_timestamp_host == IMU_HOST_STAMP
        assert reply.imu_timestamp_system == batch.imu_timestamp_system


if __name__ == "__main__":
    native_pipeline_check()

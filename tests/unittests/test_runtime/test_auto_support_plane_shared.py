"""Exercise automatic selection and refit with native host queues and Open3D."""

import subprocess
import sys
from datetime import timedelta

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import SupportPlane
from depthai_nodes.node import (
    AutoSupportPlane,
    AutoSupportPlaneConfig,
    SupportPlaneCalibration,
    SupportPlaneCalibrationConfig,
)
from tests.unittests.test_runtime.test_support_plane_calibration import (
    SYSTEM_STAMP,
    cloud_message,
    imu_message,
)

SIZE = (96, 96)


class Source(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.cloud = self.createOutput("cloud")
        self.imu = self.createOutput("imu")

    def run(self):
        pass

    def frame(self, sequence, depth):
        self.imu.send(imu_message(-9.8))
        self.cloud.send(cloud_message(sequence, SIZE, depth, system_stamp=SYSTEM_STAMP))


def receive(queue):
    result = queue.get(timedelta(seconds=5))
    assert result is not None, "Timed out waiting for pipeline output"
    return result


def run_case(mode, initial=None, *, reject_initial=False):
    calibration = SupportPlaneCalibrationConfig(vertical_distance_range_mm=(700, 1250))
    config = AutoSupportPlaneConfig(
        mode=mode,
        calibration=calibration,
        check_interval_sec=0.1,
        failed_checks_before_refit=2,
    )
    with dai.Pipeline(False) as pipeline:
        source = pipeline.create(Source)
        node = pipeline.create(AutoSupportPlane).build(
            source.cloud, source.imu, SIZE, config, initialPlane=initial
        )
        assert any(
            isinstance(n, SupportPlaneCalibration) for n in pipeline.getAllNodes()
        )
        batches = node.candidatesOutput.createOutputQueue()
        checks = node.verificationOutput.createOutputQueue()
        results = node.out.createOutputQueue()
        pipeline.start()
        if initial is None:
            source.frame(1, 1600)
            rejected = receive(batches)
            assert rejected.assessments and not any(
                a.accepted for a in rejected.assessments
            )
            assert results.tryGet() is None
            depths = np.full(SIZE[::-1], 1050)
            depths[:, :32] = 800
            source.frame(2, depths)
            batch = receive(batches)
            assert sum(a.accepted for a in batch.assessments) == 2
            result = receive(results)
            assert result.passed
            assert result.getSequenceNum() == 2
            assert result.getTimestampSystem() == SYSTEM_STAMP
            assert result.plane.offset_mm == pytest.approx(-1050)
            assert checks.tryGet() is None
            return
        source.frame(1, 1100 if reject_initial else 800)
        gate = receive(checks)
        result = receive(results)
        assert gate.request_id == 1
        if reject_initial:
            assert not gate.assessment.accepted
            assert not result.passed and result.plane is None
            batch = receive(batches)
            assert any(a.accepted for a in batch.assessments)
            replacement = receive(results)
            assert replacement.passed
            assert replacement.plane.offset_mm == pytest.approx(-1100)
            return
        assert gate.assessment.accepted and result.passed
        assert result.plane == initial
        assert batches.tryGet() is None
        if mode == AutoSupportPlaneConfig.Mode.ON_START:
            source.frame(2, 1100)
            assert checks.get(timedelta(seconds=0.3)) is None
            assert results.tryGet() is None
            return

        source.frame(2, 1100)
        for request_id in (2, 3):
            failed = receive(checks)
            assert failed.request_id == request_id
            assert failed.assessment.plane == initial
            assert not failed.assessment.accepted
        refit = receive(results)
        assert refit.passed and refit.plane.offset_mm == pytest.approx(-1100)
        assert refit.getSequenceNum() == 2
        assert any(
            a.plane == refit.plane and a.accepted for a in receive(batches).assessments
        )
        source.frame(3, 1100)
        check = receive(checks)
        assert check.request_id == 4 and check.assessment.accepted
        assert receive(results).plane == refit.plane


def test_auto_support_plane_native_pipeline():
    pytest.importorskip("open3d")
    subprocess.run([sys.executable, "-m", __name__], check=True, timeout=30)


if __name__ == "__main__":
    initial_plane = SupportPlane(normal=(0, 0, 1), offset_mm=-800)
    run_case(AutoSupportPlaneConfig.Mode.ON_START)
    run_case(AutoSupportPlaneConfig.Mode.ON_START, initial_plane)
    run_case(AutoSupportPlaneConfig.Mode.ON_START, initial_plane, reject_initial=True)
    run_case(AutoSupportPlaneConfig.Mode.CONTINUOUS, initial_plane)

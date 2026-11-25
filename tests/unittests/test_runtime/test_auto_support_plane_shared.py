"""Exercise automatic selection and refit with native host queues and Open3D."""

import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from threading import Event
from unittest.mock import patch

import depthai as dai
import numpy as np
import pytest

from depthai_nodes.message import AutoSupportPlaneConfig, SupportPlane
from depthai_nodes.node import (
    AutoSupportPlaneCalibration,
    PlaneCalibrationConfig,
    SupportPlaneCalibration,
    SupportPlaneCalibrationConfig,
)
from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode
from depthai_nodes.node.support_plane_calibrator.support_plane_fitter import (
    SupportPlaneFitter,
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
    config = AutoSupportPlaneConfig(
        mode=mode,
        vertical_distance_range_mm=(700, 1250),
        check_interval_sec=0.1,
        failed_checks_before_refit=2,
    )
    with dai.Pipeline(False) as pipeline:
        source = pipeline.create(Source)
        node = pipeline.create(AutoSupportPlaneCalibration).build(
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


def run_runtime_config_case():
    initial = SupportPlane(normal=(0, 0, 1), offset_mm=-800)
    config = AutoSupportPlaneConfig(
        calibration=SupportPlaneCalibrationConfig(
            attempt_interval_sec=0.01,
            vertical_distance_range_mm=(700, 850),
        ),
        check_interval_sec=0.01,
    )
    with dai.Pipeline(False) as pipeline:
        source = pipeline.create(Source)
        node = pipeline.create(AutoSupportPlaneCalibration).build(
            source.cloud, source.imu, SIZE, config, initialPlane=initial
        )
        configs = node.inputConfig.createInputQueue()
        batches = node.candidatesOutput.createOutputQueue()
        checks = node.verificationOutput.createOutputQueue()
        results = node.out.createOutputQueue()
        pipeline.start()
        source.frame(1, 800)
        assert receive(checks).accepted
        assert receive(results).plane == initial

        narrowed = replace(config, vertical_distance_range_mm=(1000, 1200))
        configs.send(narrowed)
        withdrawn = receive(results)
        assert not withdrawn.passed and withdrawn.plane is None
        rejected = receive(checks)
        assert rejected.plane == initial and not rejected.accepted
        assert rejected.assessment.reasons[0].code == "vertical_distance_out_of_range"
        assert not receive(results).passed
        assert not any(a.accepted for a in receive(batches).assessments)
        assert results.tryGet() is None

        source.frame(2, 1100)
        replacement = receive(results)
        assert replacement.passed
        assert replacement.plane.offset_mm == pytest.approx(-1100)

        configs.send(replace(narrowed, mode=AutoSupportPlaneConfig.Mode.CONTINUOUS))
        assert receive(checks).accepted
        assert receive(results).plane == replacement.plane


def run_legacy_runtime_config_case():
    config = AutoSupportPlaneConfig(
        calibration=PlaneCalibrationConfig(max_calibration_attempts=1),
        vertical_distance_range_mm=(700, 900),
    )
    # The legacy base class looks up a physical device during initialization.
    with (
        patch.object(
            BaseThreadedHostNode, "__init__", dai.node.ThreadedHostNode.__init__
        ),
        dai.Pipeline(False) as pipeline,
    ):
        source = pipeline.create(Source)
        node = pipeline.create(AutoSupportPlaneCalibration).build(
            source.cloud, source.imu, SIZE, config
        )
        configs = node.inputConfig.createInputQueue()
        results = node.out.createOutputQueue()
        pipeline.start()
        source.frame(1, 1000)
        rejected = receive(results)
        assert not rejected.passed and rejected.plane is None

        configs.send(replace(config, vertical_distance_range_mm=(950, 1050)))
        source.frame(2, 1000)
        result = receive(results)
        assert result.passed
        assert result.plane.offset_mm == pytest.approx(-1000)


def run_config_update_during_fit():
    fit_started = Event()
    update_applied = Event()
    release_fit = Event()
    propose = SupportPlaneFitter.propose
    apply_config = AutoSupportPlaneCalibration._apply_config

    def pause_first_fit(self, points, gravity):
        if not fit_started.is_set():
            fit_started.set()
            assert release_fit.wait(5)
        return propose(self, points, gravity)

    def observe_update(self, config):
        apply_config(self, config)
        update_applied.set()

    with (
        patch.object(SupportPlaneFitter, "propose", pause_first_fit),
        patch.object(AutoSupportPlaneCalibration, "_apply_config", observe_update),
        dai.Pipeline(False) as pipeline,
    ):
        source = pipeline.create(Source)
        node = pipeline.create(AutoSupportPlaneCalibration).build(
            source.cloud, source.imu, SIZE
        )
        configs = node.inputConfig.createInputQueue()
        batches = node.candidatesOutput.createOutputQueue()
        results = node.out.createOutputQueue()
        pipeline.start()
        try:
            source.frame(1, 800)
            assert fit_started.wait(5)
            configs.send(
                AutoSupportPlaneConfig(vertical_distance_range_mm=(1000, 1200))
            )
            assert update_applied.wait(5)
        finally:
            release_fit.set()

        assert any(a.accepted for a in receive(batches).assessments)
        source.frame(2, 1100)
        result = receive(results)
        assert result.getSequenceNum() == 2
        assert result.plane.offset_mm == pytest.approx(-1100)


def test_auto_support_plane_native_pipeline():
    pytest.importorskip("open3d")
    subprocess.run([sys.executable, "-m", __name__], check=True, timeout=45)


if __name__ == "__main__":
    initial_plane = SupportPlane(normal=(0, 0, 1), offset_mm=-800)
    run_case(AutoSupportPlaneConfig.Mode.ON_START)
    run_case(AutoSupportPlaneConfig.Mode.ON_START, initial_plane)
    run_case(AutoSupportPlaneConfig.Mode.ON_START, initial_plane, reject_initial=True)
    run_case(AutoSupportPlaneConfig.Mode.CONTINUOUS, initial_plane)
    run_runtime_config_case()
    run_legacy_runtime_config_case()
    run_config_update_during_fit()

import json
import struct
import threading
import time

import depthai as dai
import numpy as np
import pytest
from controller import CLOUD_SIZE, Controller, default_settings

from depthai_nodes.message import (
    AutoSupportPlaneResult,
    CalibrationOutcome,
    SupportPlane,
    SupportPlaneCalibrationResult,
    SupportPlaneCandidate,
)


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for selector"
        time.sleep(0.001)


@pytest.fixture
def controller(tmp_path):
    instance = Controller(tmp_path / "plane.json")
    instance.begin(default_settings())
    yield instance
    instance.close()


def candidates():
    return tuple(
        SupportPlaneCandidate(
            plane=SupportPlane(normal=(0, 0, 1), offset_mm=-depth),
            inlier_count=100,
            inlier_ratio=0.5,
        )
        for depth in (1000, 1500)
    )


def start_selection(controller, choices):
    returned = []
    thread = threading.Thread(
        target=lambda: returned.append(
            controller.select_candidate(choices, controller.epoch)
        ),
        daemon=True,
    )
    thread.start()
    wait_for(lambda: controller.snapshot()["waiting"])
    return thread, returned


def command(controller, op, **kwargs):
    return controller.control(
        {"op": op, "ticket": controller.ticket, "epoch": controller.epoch, **kwargs}
    )


def cloud(sequence, z):
    points = np.zeros((CLOUD_SIZE[0] * CLOUD_SIZE[1], 3), dtype=np.float32)
    points[:, 2] = z
    result = dai.PointCloudData()
    result.setPoints(points)
    result.setSequenceNum(sequence)
    return result


def decode(data):
    length = struct.unpack_from("<I", data)[0]
    return json.loads(data[4 : 4 + length]), np.frombuffer(
        data[4 + length :], dtype="<f4"
    )


def test_preview_then_confirm_returns_the_exact_candidate_and_rejects_duplicate(
    controller,
):
    choices = candidates()
    thread, returned = start_selection(controller, choices)
    assert command(controller, "preview", index=1)["ok"]
    assert controller.snapshot()["preview_index"] == 1
    assert command(controller, "confirm", index=1)["ok"]
    assert not command(controller, "confirm", index=0)["ok"]
    thread.join(2)
    assert returned[0] is choices[1]


@pytest.mark.parametrize("index", [True, "1", 1.0, -1, 2])
def test_invalid_candidate_index_does_not_complete_selection(controller, index):
    thread, returned = start_selection(controller, candidates())
    assert not command(controller, "confirm", index=index)["ok"]
    assert returned == []
    controller.close()
    thread.join(2)


def test_retry_returns_none_and_a_previous_ticket_cannot_confirm_new_candidates(
    controller,
):
    choices = candidates()
    thread, returned = start_selection(controller, choices)
    old_ticket = controller.ticket
    assert command(controller, "reject")["ok"]
    thread.join(2)
    assert returned == [None]
    thread, returned = start_selection(controller, candidates())
    assert not controller.control(
        {"op": "confirm", "epoch": controller.epoch, "ticket": old_ticket, "index": 0}
    )["ok"]
    assert returned == []
    controller.close()
    thread.join(2)


def test_restart_cancels_wait_even_if_main_consumes_request_before_selector_wakes(
    controller,
):
    thread, returned = start_selection(controller, candidates())
    with controller.condition:
        assert command(controller, "restart")["ok"]
        settings, initial = controller.take_restart()
        assert initial is None
    thread.join(2)
    assert not thread.is_alive()
    assert returned == [None]
    controller.begin(settings)
    assert not controller.snapshot()["waiting"]


@pytest.mark.parametrize("telemetry_first", [True, False])
def test_preview_freezes_the_exact_fitted_cloud_despite_new_live_frames(
    controller, telemetry_first
):
    result = SupportPlaneCalibrationResult(CalibrationOutcome.SUCCEEDED, candidates())
    result.setSequenceNum(7)
    controller.record_cloud(cloud(7, 1000))
    if telemetry_first:
        controller.record_calibration(result)
    thread, returned = start_selection(controller, result.candidates)
    if not telemetry_first:
        controller.record_calibration(result)
    for sequence in range(8, 50):
        controller.record_cloud(cloud(sequence, 2000))
    metadata, xyz = decode(controller.cloud_bytes())
    assert metadata["frozen"] and metadata["sequence"] == 7
    assert np.all(xyz[2::3] == 1000)
    command(controller, "confirm", index=0)
    thread.join(2)
    controller.record_result(AutoSupportPlaneResult(True, returned[0].plane))
    metadata, xyz = decode(controller.cloud_bytes())
    assert not metadata["frozen"] and metadata["sequence"] == 49
    assert np.all(xyz[2::3] == 2000)


def test_identical_geometry_in_a_new_attempt_never_reuses_the_old_fitting_frame(
    controller,
):
    old = SupportPlaneCalibrationResult(CalibrationOutcome.SUCCEEDED, candidates())
    old.setSequenceNum(7)
    controller.record_calibration(old)
    thread, _ = start_selection(controller, candidates())
    metadata, _ = decode(controller.cloud_bytes())
    assert metadata["sequence"] is None and metadata["count"] == 0
    controller.close()
    thread.join(2)


def test_confirmed_plane_persists_and_failed_refit_keeps_it(controller):
    plane = candidates()[0].plane
    controller.record_result(AutoSupportPlaneResult(True, plane))
    assert SupportPlane.fromJsonFile(controller.saved_path) == plane
    controller.record_result(AutoSupportPlaneResult(False, plane))
    state = controller.snapshot()
    assert (
        state["held_plane"] == plane.toJson()
        and state["last_result"]["passed"] is False
    )
    assert command(controller, "export")["plane"] == plane.toJson()


def test_restore_is_a_gated_restart_and_invalid_json_preserves_current_session(
    controller,
):
    assert not command(
        controller, "restore", plane={"normal": [0, 0, 0], "offset_mm": 1}
    )["ok"]
    assert controller.take_restart() is None
    plane = candidates()[0].plane
    assert command(controller, "restore", plane=plane.toJson())["ok"]
    settings, initial = controller.take_restart()
    assert initial == plane
    controller.begin(settings, initial)
    assert controller.snapshot()["phase"] == "verifying restored plane"
    assert not command(controller, "export")["ok"]


def test_accepted_import_replaces_a_different_previously_saved_plane(controller):
    old, restored = candidates()[0].plane, candidates()[1].plane
    controller.record_result(AutoSupportPlaneResult(True, old))
    controller.begin(default_settings(), restored)
    controller.record_result(AutoSupportPlaneResult(True, restored))
    assert SupportPlane.fromJsonFile(controller.saved_path) == restored


def test_invalid_settings_do_not_restart_and_uncapped_search_is_supported(controller):
    settings = default_settings()
    settings["calibration"]["max_calibration_attempts"] = None
    assert command(controller, "restart", settings=settings)["ok"]
    controller.take_restart()
    settings["calibration"]["min_on_plane_ratio"] = 2
    assert not command(controller, "restart", settings=settings)["ok"]
    assert controller.take_restart() is None


def test_bad_commands_are_errors_not_exceptions(controller):
    for request in (None, [], {}, {"op": "bad"}, {"op": "preview", "index": -1}):
        assert controller.control(request)["ok"] is False

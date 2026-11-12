"""Native host-only integration checks; no device discovery or hardware access."""

import json
import struct
import time

import pytest
import serve_simulation as simulation
from controller import Controller, default_settings
from main import drain_telemetry

from depthai_nodes.message import SupportPlane

pytest.importorskip("open3d")


def pump(queues, controller, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        drain_telemetry(queues, controller)
        if predicate(controller.snapshot()):
            return controller.snapshot()
        time.sleep(0.01)
    pytest.fail(f"Native node timed out: {controller.snapshot()}")


def confirm(controller, index=0):
    state = controller.snapshot()
    result = controller.control(
        {
            "op": "confirm",
            "epoch": state["epoch"],
            "ticket": state["ticket"],
            "index": index,
        }
    )
    assert result["ok"], result


def shutdown(pipeline, controller):
    controller.close()
    pipeline.stop()
    pipeline.wait()


def test_native_manual_selection_movement_refit_and_occlusion(tmp_path):
    controller = Controller(tmp_path / "plane.json")
    settings = default_settings()
    settings.update(check_interval_sec=0.05, failed_checks_before_refit=1)
    settings["calibration"]["max_calibration_attempts"] = 2
    scene = simulation.Scene()
    pipeline, queues = simulation.construct(controller, settings, None, scene)
    pipeline.start()
    try:
        first = pump(queues, controller, lambda s: s["waiting"])
        assert len(first["candidates"]) == 2
        # Telemetry associates the exact fitted cloud through real native fan-out.
        pump(queues, controller, lambda _: controller.candidate_sequence is not None)
        data = controller.cloud_bytes()
        length = struct.unpack_from("<I", data)[0]
        metadata = json.loads(data[4 : 4 + length])
        assert metadata["frozen"] and metadata["count"] == 16000
        confirm(controller, 1)
        held = pump(queues, controller, lambda s: s["phase"] == "holding")
        assert held["held_plane"]["offset_mm"] == pytest.approx(-1550, abs=1)
        pump(
            queues,
            controller,
            lambda s: bool(s["verification"] and s["verification"]["accepted"]),
        )

        scene.control({"op": "move"})
        changed = pump(
            queues, controller, lambda s: s["waiting"] and s["ticket"] > first["ticket"]
        )
        assert changed["verification"]["accepted"] is False
        assert changed["held_plane"] == held["held_plane"]
        assert [
            c["plane"]["offset_mm"] for c in changed["candidates"]
        ] == pytest.approx([-1300, -1850], abs=1)
        confirm(controller)
        pump(
            queues,
            controller,
            lambda s: s["phase"] == "holding" and s["held_plane"]["offset_mm"] < -1200,
        )

        scene.control({"op": "occlude"})
        failed = pump(
            queues,
            controller,
            lambda s: bool(s["last_result"] and not s["last_result"]["passed"]),
        )
        assert failed["held_plane"]["offset_mm"] == pytest.approx(-1300, abs=1)
        assert failed["calibration_outcome"] == "failed"
    finally:
        shutdown(pipeline, controller)


@pytest.mark.parametrize("offset,accepted", [(-1000, True), (-3000, False)])
def test_restored_plane_is_verified_before_acceptance_or_new_search(
    tmp_path, offset, accepted
):
    controller = Controller(tmp_path / "plane.json")
    settings = default_settings()
    settings["mode"] = "on_start"
    initial = SupportPlane(normal=(0, 0, 1), offset_mm=offset)
    pipeline, queues = simulation.construct(
        controller, settings, initial, simulation.Scene()
    )
    pipeline.start()
    try:
        state = pump(queues, controller, lambda s: s["last_result"] is not None)
        assert state["last_result"]["passed"] is accepted
        assert bool(state["last_result"]["plane"]) is accepted
        if accepted:
            assert state["candidates"] == []
            assert state["calibration_outcome"] is None
        else:
            pump(queues, controller, lambda s: s["waiting"])
    finally:
        shutdown(pipeline, controller)


def test_automatic_on_start_uses_default_largest_candidate_and_never_rechecks(tmp_path):
    controller = Controller(tmp_path / "plane.json")
    settings = default_settings()
    settings.update(mode="on_start", selection="automatic", check_interval_sec=0.05)
    scene = simulation.Scene()
    pipeline, queues = simulation.construct(controller, settings, None, scene)
    pipeline.start()
    try:
        state = pump(queues, controller, lambda s: s["phase"] == "holding")
        assert state["held_plane"]["offset_mm"] == pytest.approx(-1000, abs=1)
        assert not state["waiting"]
        scene.control({"op": "move"})
        state = pump(queues, controller, lambda _: controller.latest_frame >= 10)
        assert state["verification"] is None
        assert state["held_plane"]["offset_mm"] == pytest.approx(-1000, abs=1)
    finally:
        shutdown(pipeline, controller)

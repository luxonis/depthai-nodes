"""Regression: restoring/reconfiguring must not reuse a device closed by stop."""

import sys
from unittest.mock import MagicMock

import main as application


def test_restore_reopens_device_after_pipeline_closes_it(monkeypatch, tmp_path):
    monkeypatch.setenv("OAKAGENT_STORAGE_APP", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(application.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(application, "start_frontend", lambda *_args: MagicMock())
    monkeypatch.setattr(
        application.dai, "RemoteConnection", lambda **_kwargs: MagicMock()
    )
    devices, pipelines, initials = [], [], []

    class Device:
        def __init__(self, *_args):
            self.closed = False
            devices.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def getPlatform(self):
            assert not self.closed, "Reused a closed camera handle"
            return application.dai.Platform.RVC4

        def getConnectedIMU(self):
            return "test accelerometer"

    class Pipeline:
        def __init__(self, device):
            assert (
                not device.closed
            ), "Reused a closed device to build the next pipeline"
            self.device = device
            pipelines.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stop()

        def start(self):
            pass

        def isRunning(self):
            return len(pipelines) == 1

        def processTasks(self):
            pass

        def stop(self):
            self.device.closed = True

    def attach(_pipeline, _cloud, _imu, controller, settings, initial):
        controller.begin(settings, initial)
        initials.append(initial)
        return None, {}

    def trigger_restore(_queues, controller):
        response = controller.control(
            {"op": "restore", "plane": {"normal": [0, 0, 1], "offset_mm": -1000}}
        )
        assert response["ok"]

    monkeypatch.setattr(application.dai, "Device", Device)
    monkeypatch.setattr(application.dai, "Pipeline", Pipeline)
    monkeypatch.setattr(
        application, "build_camera_sources", lambda *_args: (None, None)
    )
    monkeypatch.setattr(application, "attach_auto_node", attach)
    monkeypatch.setattr(application, "drain_telemetry", trigger_restore)
    application.main()
    assert len(devices) == 2 and devices[0] is not devices[1]
    assert all(device.closed for device in devices)
    assert initials[0] is None and initials[1].offset_mm == -1000

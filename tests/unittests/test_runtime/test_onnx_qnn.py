import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from depthai_nodes.runtime import onnx_qnn, onnx_qnn_session


class FakeSessionOptions:
    def __init__(self):
        self.config_entries = []
        self.provider_devices = []

    def add_session_config_entry(self, key, value):
        self.config_entries.append((key, value))

    def add_provider_for_devices(self, devices, options):
        self.provider_devices.append((devices, options))


class FakeSession:
    def __init__(self):
        self.fallback_disabled = False

    def disable_fallback(self):
        self.fallback_disabled = True


class FakeOrt:
    SessionOptions = FakeSessionOptions
    __version__ = "1.0.0"

    def __init__(self, devices):
        self.devices = devices
        self.registered_libraries = []
        self.sessions = []

    def register_execution_provider_library(self, name, path):
        self.registered_libraries.append((name, path))

    def get_ep_devices(self):
        return self.devices

    def InferenceSession(self, *args, **kwargs):
        session = FakeSession()
        self.sessions.append((args, kwargs, session))
        return session


@pytest.fixture(autouse=True)
def reset_onnx_qnn_registration():
    onnx_qnn._registered = False
    yield
    onnx_qnn._registered = False


def test_onnx_qnn_session_falls_back_to_cpu_without_qnn_plugin(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    ort = FakeOrt([])
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    real_import = builtins.__import__

    def import_without_qnn(name, *args, **kwargs):
        if name == "onnxruntime_qnn":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_qnn)
    logger = MagicMock()
    monkeypatch.setattr(onnx_qnn, "logger", logger)

    onnx_qnn_session(model_path, device_wait_s=0)

    args, kwargs, _ = ort.sessions[0]
    assert args[0] == str(model_path)
    assert isinstance(args[1], FakeSessionOptions)
    assert kwargs == {"providers": ["CPUExecutionProvider"]}
    logger.warning.assert_called_once_with(
        "QNN EP unavailable (%s); falling back to CPU EP",
        "onnxruntime_qnn is not installed (use the onnxruntime variant "
        "of the oakapp-base image)",
    )


def test_onnx_qnn_session_registers_qnn_provider_and_disables_runtime_fallback(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    device = SimpleNamespace(ep_name="QNNExecutionProvider")
    ort = FakeOrt([device])
    qnn_plugin = ModuleType("onnxruntime_qnn")
    qnn_plugin.__version__ = "1.0.0"
    qnn_plugin.get_library_path = lambda: "/qnn/provider.so"
    qnn_plugin.get_qnn_htp_path = lambda: "/qnn/htp.so"
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", qnn_plugin)

    session = onnx_qnn_session(
        model_path,
        cache_context=False,
        ep_options={"foo": 1},
        fallback_to_cpu=False,
        runtime_fallback="raise",
    )

    assert ort.registered_libraries == [("QNNExecutionProvider", "/qnn/provider.so")]
    args, kwargs, created_session = ort.sessions[0]
    assert args == (str(model_path),)
    assert set(kwargs) == {"sess_options"}
    options = kwargs["sess_options"]
    assert options.config_entries == [("session.disable_cpu_ep_fallback", "1")]
    assert options.provider_devices == [
        (
            [device],
            {
                "backend_path": "/qnn/htp.so",
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1",
                "foo": "1",
            },
        )
    ]
    assert session is created_session
    assert session.fallback_disabled

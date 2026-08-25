"""Create ONNX Runtime sessions on the OAK4 Hexagon DSP."""

import hashlib
import math
import os
import tempfile
import time
from pathlib import Path

from depthai_nodes.logging import get_logger

_EP_NAME = "QNNExecutionProvider"
_registered = False
logger = get_logger(__name__)


def onnx_qnn_session(
    model_path,
    *,
    fp16=True,
    cache_context=True,
    performance_mode="burst",
    fallback_to_cpu=True,
    runtime_fallback="ort",
    device_wait_s=60,
    ep_options=None,
    session_options=None,
    verbose=False,
):
    """Create an ONNX Runtime session running on the OAK4 DSP (HTP).

    This helper is intended for the ``onnxruntime`` variant of ``oakapp-base``.
    That image provides the FastRPC setup, QNN plugin, and required device nodes.

    Args:
        model_path: path to a .onnx model. Inputs must have static shapes.
        fp16: run fp32 graphs in fp16 on the HTP (no-op for QDQ int8 models).
        cache_context: cache the compiled QNN graph (EPContext) next to the
            model so subsequent session creations skip HTP graph compilation.
        performance_mode: QNN HTP performance mode (for example, ``burst``).
        fallback_to_cpu: if False, raise when the DSP is unavailable during
            session creation or any node cannot be placed on it. If True,
            return a CPU session when no QNN device is available.
        runtime_fallback: behavior after a QNN execution error. ``"ort"``
            keeps ONNX Runtime's automatic fallback behavior; ``"raise"``
            disables it so the caller can handle the error explicitly.
        device_wait_s: seconds to wait for a QNN device to appear during
            startup. Set to 0 to check once.
        ep_options: extra QNN EP provider options (dict), merged last.
        session_options: pre-configured ort.SessionOptions to extend.
        verbose: enable verbose ORT logging.

    Returns:
        onnxruntime.InferenceSession
    """
    import onnxruntime as ort

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if runtime_fallback not in {"ort", "raise"}:
        raise ValueError("runtime_fallback must be 'ort' or 'raise'")
    try:
        device_wait_s = float(device_wait_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("device_wait_s must be a non-negative number") from exc
    if not math.isfinite(device_wait_s) or device_wait_s < 0:
        raise ValueError("device_wait_s must be a non-negative number")

    so = session_options or ort.SessionOptions()
    if verbose:
        so.log_severity_level = 0

    qnn_devs, why_not = _qnn_ep_devices(ort, device_wait_s)
    if not qnn_devs:
        if not fallback_to_cpu:
            raise RuntimeError(f"QNN EP unavailable: {why_not}")
        logger.warning("QNN EP unavailable (%s); falling back to CPU EP", why_not)
        return ort.InferenceSession(
            str(model_path), so, providers=["CPUExecutionProvider"]
        )

    import onnxruntime_qnn

    opts = {
        "backend_path": onnxruntime_qnn.get_qnn_htp_path(),
        "htp_performance_mode": performance_mode,
        "enable_htp_fp16_precision": "1" if fp16 else "0",
    }
    if ep_options:
        opts.update({key: str(value) for key, value in ep_options.items()})

    if not fallback_to_cpu:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    load_path = model_path
    if cache_context:
        ctx_path = _context_cache_path(model_path, opts)
        if ctx_path.is_file():
            load_path = ctx_path
        else:
            so.add_session_config_entry("ep.context_enable", "1")
            so.add_session_config_entry("ep.context_file_path", str(ctx_path))

    so.add_provider_for_devices(qnn_devs, opts)
    session = ort.InferenceSession(str(load_path), sess_options=so)
    if runtime_fallback == "raise":
        session.disable_fallback()
    return session


def _qnn_ep_devices(ort, device_wait_s):
    """Register the plugin EP and return its devices and an error reason."""
    global _registered
    try:
        import onnxruntime_qnn
    except ImportError:
        return [], (
            "onnxruntime_qnn is not installed (use the onnxruntime variant "
            "of the oakapp-base image)"
        )
    if not _registered:
        ort.register_execution_provider_library(
            _EP_NAME, onnxruntime_qnn.get_library_path()
        )
        _registered = True
    deadline = time.monotonic() + device_wait_s
    while True:
        devices = [
            device for device in ort.get_ep_devices() if device.ep_name == _EP_NAME
        ]
        if devices:
            return devices, ""
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return [], (
                "no NPU EP device was enumerated after "
                f"{device_wait_s:g}s; make sure the app runs on the "
                "onnxruntime variant of the oakapp-base image with "
                "/dev/fastrpc-cdsp and the dma_heap nodes in optional_devices, "
                "and routes through "
                "/entrypoint.sh"
            )
        time.sleep(min(1.0, remaining_s))


def _context_cache_path(model_path, opts):
    """Return the deterministic EPContext cache path for a model and options."""
    import onnxruntime as ort
    import onnxruntime_qnn

    stat = model_path.stat()
    key = "|".join(
        [
            str(model_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            ort.__version__,
            onnxruntime_qnn.__version__,
            str(sorted(opts.items())),
        ]
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]

    cache_dir = os.environ.get("OAK4ORT_CACHE_DIR")
    if cache_dir:
        cache_dir = Path(cache_dir)
    elif os.access(model_path.parent, os.W_OK):
        cache_dir = model_path.parent / ".oak4ort_cache"
    else:
        cache_dir = Path(tempfile.gettempdir()) / "oak4ort_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_path.stem}-{digest}_ctx.onnx"

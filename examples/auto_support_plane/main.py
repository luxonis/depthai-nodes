"""OAK4 standalone AutoSupportPlane showcase; see README.md for run commands."""

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path

import depthai as dai

try:
    from depthai_nodes.message import SupportPlane
    from depthai_nodes.node import AutoSupportPlane, SupportPlaneCalibrator
except ImportError as exc:
    raise SystemExit(
        "Install this example's pinned requirements.txt (PR #340)"
    ) from exc

from controller import CLOUD_SIZE, Controller, default_settings, make_config
from frontend_server import start_frontend

logger = logging.getLogger("auto-support-plane")


def attach_auto_node(pipeline, cloud_output, imu_output, controller, settings, initial):
    """Only public node APIs are used, including observing the calibrator subnode."""
    epoch = controller.begin(settings, initial)
    selector = None
    if settings["selection"] == "manual":

        def selector(candidates):
            return controller.select_candidate(candidates, epoch)

    node = pipeline.create(AutoSupportPlane).build(
        cloud_output,
        imu_output,
        CLOUD_SIZE,
        config=make_config(settings),
        initialPlane=initial,
        selectCandidate=selector,
    )
    # Observe results without taking over the wrapper's controls. Every output
    # fan-out gets its own queue, so the wrapper still receives every message.
    calibrator = next(
        n for n in pipeline.getAllNodes() if isinstance(n, SupportPlaneCalibrator)
    )
    return node, {
        "cloud": cloud_output.createOutputQueue(maxSize=16, blocking=False),
        "imu": imu_output.createOutputQueue(maxSize=4, blocking=False),
        "calibration": calibrator.out.createOutputQueue(maxSize=4, blocking=False),
        "verification": calibrator.verificationOutput.createOutputQueue(
            maxSize=4, blocking=False
        ),
        "result": node.out.createOutputQueue(maxSize=4, blocking=False),
    }


def drain_telemetry(queues, controller):
    handlers = {
        "cloud": controller.record_cloud,
        "imu": controller.record_imu,
        "calibration": controller.record_calibration,
        "verification": controller.record_verification,
        "result": controller.record_result,
    }
    for name, handler in handlers.items():
        while (message := queues[name].tryGet()) is not None:
            handler(message)


def build_camera_sources(pipeline, fps):
    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color = camera.requestOutput(
        CLOUD_SIZE,
        type=dai.ImgFrame.Type.RGB888i,
        fps=fps,
        enableUndistortion=True,
    )
    depth = pipeline.create(dai.node.Depth)
    depth.build(dai.node.Depth.Algorithm.AUTO, size=CLOUD_SIZE, fps=fps)
    depth.setAlignTo(color)
    rgbd = pipeline.create(dai.node.RGBD).build()
    color.link(rgbd.inColor)
    depth.depth.link(rgbd.inDepth)

    imu = pipeline.create(dai.node.IMU)
    # RAW uses sensor-native axes. UNCALIBRATED is the SDK's processed
    # accelerometer in Luxonis RDF (right/down/forward), without requiring
    # factory intrinsic correction. Do not apply another rotation to it.
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_UNCALIBRATED, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    return rgbd.pcl, imu.out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="Device IP or MXID for a peripheral host run")
    parser.add_argument("--bind", default="0.0.0.0")
    # oakapp-base's nginx already owns OAKAPP_STATIC_FRONTEND_PORT. The
    # custom Python HTTP server must use a separate port.
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument(
        "--initial-plane", type=Path, help="Verify this saved plane on startup"
    )
    parser.add_argument(
        "--no-restore", action="store_true", help="Ignore persisted plane"
    )
    args = parser.parse_args()
    if not 0 < args.fps <= 30:
        parser.error("--fps must be greater than 0 and at most 30")
    logging.basicConfig(level=logging.INFO)
    directory = Path(__file__).resolve().parent
    storage = Path(os.getenv("OAKAGENT_STORAGE_APP", str(directory / "storage")))
    controller = Controller(storage / "support_plane.json")
    initial = None
    initial_path = args.initial_plane or (
        None if args.no_restore else controller.saved_path
    )
    if initial_path is not None and initial_path.exists():
        initial = SupportPlane.fromJsonFile(initial_path)
    elif args.initial_plane is not None:
        parser.error(f"Initial plane not found: {args.initial_plane}")

    stopping = threading.Event()

    def stop(_signum, _frame):
        stopping.set()
        controller.close()  # Wake a selector before stopping the pipeline.

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    remote = dai.RemoteConnection(
        address=args.bind, webSocketPort=args.ws_port, serveFrontend=False
    )
    remote.registerService("plane_control", controller.control)
    remote.registerBinaryService("plane_cloud", controller.cloud_bytes)
    frontend = Path(
        os.getenv("OAKAPP_STATIC_FRONTEND_PATH", str(directory / "frontend"))
    )
    server = start_frontend(frontend, args.bind, args.port, args.ws_port)
    logger.info(
        "Viewer: http://<OAK4-IP>:%s (host run: http://localhost:%s)",
        args.port,
        args.port,
    )
    settings = default_settings()
    try:
        while not stopping.is_set():
            # Pipeline.__exit__ closes its device in DepthAI 3.12. A rebuild
            # must create a fresh handle rather than reuse that closed one.
            device = (
                dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
            )
            with device:
                if device.getPlatform() != dai.Platform.RVC4:
                    raise RuntimeError(
                        "This standalone showcase requires an OAK4 / RVC4 device"
                    )
                if not device.getConnectedIMU():
                    raise RuntimeError(
                        "An accelerometer is required for plane calibration"
                    )
                with dai.Pipeline(device) as pipeline:
                    cloud, imu = build_camera_sources(pipeline, args.fps)
                    _, queues = attach_auto_node(
                        pipeline, cloud, imu, controller, settings, initial
                    )
                    remote.addTopic("organized_cloud", cloud)
                    pipeline.start()
                    remote.registerPipeline(pipeline)
                    request = None
                    try:
                        while pipeline.isRunning() and not stopping.is_set():
                            pipeline.processTasks()
                            drain_telemetry(queues, controller)
                            request = controller.take_restart()
                            if request is not None:
                                break
                            time.sleep(0.01)
                    finally:
                        # Also wakes selection if the native pipeline stops itself.
                        with controller.condition:
                            controller.epoch += 1
                            controller.condition.notify_all()
                        pipeline.stop()
                        remote.removeTopic("organized_cloud")
            if request is None:
                break
            settings, initial = request
    except Exception as exc:
        logger.exception("Application failed")
        with controller.condition:
            controller.data.update(phase="error", error=str(exc))
        raise
    finally:
        controller.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

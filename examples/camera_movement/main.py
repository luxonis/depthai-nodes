"""Show camera movement on an H.264 stream from a standalone OAK4."""

import math
import signal
import time
from collections import deque
from dataclasses import asdict, replace
from datetime import timedelta
from threading import Event, Lock

import cv2
import depthai as dai

from depthai_nodes.message import CameraMovementControl
from depthai_nodes.node import CameraMovementConfig, CameraMovementDetector

SIZE = (1280, 720)
FPS = 30
ROTATION_RATE = math.radians(3)
GYRO_MAX_AGE = 0.1
DEMO_CONFIG = CameraMovementConfig(turn_threshold_deg=15)


def draw_moving(frame: dai.ImgFrame) -> None:
    """Draw on the NV12 luma plane and preserve the frame metadata."""
    data = frame.getData().copy()
    width, height, stride = frame.getWidth(), frame.getHeight(), frame.getStride()
    luma = data[: height * stride].reshape(height, stride)[:, :width]
    for color, thickness in ((16, 8), (235, 3)):
        cv2.putText(
            luma,
            "Moving",
            (24, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            color,
            thickness,
            cv2.LINE_AA,
        )
    frame.setData(data)


class MovementOverlay(dai.node.ThreadedHostNode):
    """Mark live rotation on video."""

    def __init__(self) -> None:
        super().__init__()
        self.video = self.createInput(name="video", queueSize=2, blocking=False)
        self.imu = self.createInput(name="imu", queueSize=8, blocking=False)
        self.out = self.createOutput(name="out")

    def run(self) -> None:
        last_gyro_stamp = timedelta()
        last_gyro_at = -math.inf
        rotating = False
        try:
            while self.isRunning():
                frame = self.video.get()
                now = time.monotonic()
                for message in self.imu.tryGetAll():
                    for packet in message.packets:
                        gyro = packet.gyroscope
                        stamp = gyro.getTimestampDevice()
                        if stamp > last_gyro_stamp:
                            last_gyro_stamp = stamp
                            last_gyro_at = now
                            rotating = (
                                math.hypot(gyro.x, gyro.y, gyro.z) >= ROTATION_RATE
                            )
                if rotating and now - last_gyro_at < GYRO_MAX_AGE:
                    draw_moving(frame)
                self.out.send(frame)
        except dai.MessageQueue.QueueException:
            return


class DetectorControls:
    """Publish detector settings, heartbeat messages, and recent alerts."""

    def __init__(self, detector: CameraMovementDetector) -> None:
        self.detector = detector
        self.control = detector.inputControl.createInputQueue()
        self.status_queue = detector.status.createOutputQueue(maxSize=4, blocking=False)
        self.event_queue = detector.out.createOutputQueue(maxSize=16, blocking=False)
        self.lock = Lock()
        self.ready = Event()
        self.pending = None
        self.status = None
        self.events = deque(maxlen=12)

    def request(self, request):
        """Handle a dashboard command and return the current state."""
        with self.lock:
            try:
                action = request.get("action")
                config = self.detector.getConfig()
                if self.pending == config:
                    self.pending = None
                if action == "apply":
                    if not self.ready.is_set() or self.pending is not None:
                        raise ValueError(
                            "Wait for the detector to finish applying settings."
                        )
                    settings = request.get("config")
                    if not isinstance(settings, dict):
                        raise ValueError("Expected detector settings.")
                    requested = replace(config, **settings)
                    if requested != config:
                        self.pending = requested
                        self.detector.setConfig(requested)
                        if requested.status_interval_s != config.status_interval_s:
                            self.status = None
                elif action == "calibrate":
                    if not self.ready.is_set() or self.pending is not None:
                        raise ValueError("Wait for the detector to be ready.")
                    if not self.control.trySend(CameraMovementControl.calibrate()):
                        raise ValueError("The calibration queue is busy. Try again.")
                    print("Calibration requested. Keep the camera still.")
                elif action != "state":
                    raise ValueError("Unknown detector command.")
            except (AttributeError, OverflowError, TypeError, ValueError) as exc:
                return {"error": str(exc)}
            return {
                "ready": self.ready.is_set(),
                "applying": self.pending is not None,
                "config": asdict(config),
                "defaults": asdict(DEMO_CONFIG),
                "status": self.status if config.status_interval_s is not None else None,
                "events": list(self.events),
            }

    def poll(self) -> None:
        """Collect status and movement messages from the running detector."""
        statuses = self.status_queue.tryGetAll()
        events = self.event_queue.tryGetAll()
        with self.lock:
            if statuses:
                status = asdict(statuses[-1])
                if (
                    self.status is None
                    or status["calibration"] != self.status["calibration"]
                ):
                    print(f"Calibration: {status['calibration'].value}")
                self.status = status
            for event in events:
                self.events.appendleft(
                    {
                        "reason": event.reason.value,
                        "angle_deg": event.angle_deg,
                        "time": time.time(),
                    }
                )
                print(f"{event.reason.value}: {event.angle_deg:.1f}°")


def main() -> None:
    stopping = Event()

    def stop(_signum, _frame):
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    visualizer = dai.RemoteConnection(address="127.0.0.1", serveFrontend=False)
    with dai.Pipeline() as pipeline:
        camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        video = camera.requestOutput(SIZE, type=dai.ImgFrame.Type.NV12, fps=FPS)

        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(
            [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100
        )
        imu.setBatchReportThreshold(4)
        imu.setMaxBatchReports(20)

        detector = pipeline.create(CameraMovementDetector).build(imu.out, DEMO_CONFIG)
        controller = DetectorControls(detector)
        visualizer.registerService("detector", controller.request)

        overlay = pipeline.create(MovementOverlay)
        video.link(overlay.video)
        imu.out.link(overlay.imu)
        encoder = pipeline.create(dai.node.VideoEncoder).build(
            overlay.out,
            frameRate=FPS,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )
        visualizer.addTopic("RGB", encoder.out)

        pipeline.start()
        controller.ready.set()
        print("Camera movement detector ready. Calibrate from the dashboard.")
        while pipeline.isRunning() and not stopping.wait(0.025):
            controller.poll()


if __name__ == "__main__":
    main()

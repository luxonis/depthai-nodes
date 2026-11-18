"""Show camera movement on an H.264 stream from a standalone OAK4."""

import signal
import time
from threading import Event

import cv2
import depthai as dai

from depthai_nodes.message import CameraMovementControl, CameraMovementEvent
from depthai_nodes.node import CameraMovementConfig, CameraMovementDetector

SIZE = (1280, 720)
FPS = 30
LABEL_SECONDS = 2.0


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
    """Stamp the next two seconds of video after each movement event."""

    def __init__(self) -> None:
        super().__init__()
        self.video = self.createInput(name="video", queueSize=2, blocking=False)
        self.movement = self.createInput(name="movement", queueSize=8, blocking=False)
        self.out = self.createOutput(name="out")

    def run(self) -> None:
        visible_until = 0.0
        try:
            while self.isRunning():
                frame = self.video.get()
                now = time.monotonic()
                for event in self.movement.tryGetAll():
                    if isinstance(event, CameraMovementEvent):
                        visible_until = now + LABEL_SECONDS
                        print(f"{event.reason.value}: {event.angle_deg:.1f}°")
                if now < visible_until:
                    draw_moving(frame)
                self.out.send(frame)
        except dai.MessageQueue.QueueException:
            return


def main() -> None:
    stopping = Event()

    def stop(_signum, _frame):
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    visualizer = dai.RemoteConnection(address="127.0.0.1", httpPort=8082)
    with dai.Pipeline(dai.Device(dai.DeviceInfo.local())) as pipeline:
        camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        video = camera.requestOutput(SIZE, type=dai.ImgFrame.Type.NV12, fps=FPS)

        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(
            [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100
        )
        imu.setBatchReportThreshold(4)
        imu.setMaxBatchReports(20)

        detector = pipeline.create(CameraMovementDetector).build(
            imu.out, CameraMovementConfig(turn_threshold_deg=15)
        )
        control = detector.inputControl.createInputQueue()
        status = detector.status.createOutputQueue(maxSize=4, blocking=False)

        overlay = pipeline.create(MovementOverlay)
        video.link(overlay.video)
        detector.out.link(overlay.movement)
        encoder = pipeline.create(dai.node.VideoEncoder).build(
            overlay.out,
            frameRate=FPS,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )
        visualizer.addTopic("RGB", encoder.out)

        pipeline.start()
        visualizer.registerPipeline(pipeline)
        print(
            "Open the OAK Visualizer and press c with the camera at rest to calibrate."
        )
        last_calibration = None
        while pipeline.isRunning() and not stopping.is_set():
            if visualizer.waitKey(50) in (ord("c"), ord("C")):
                control.send(CameraMovementControl.calibrate())
                print("Calibration requested. Keep the camera still.")
            for message in status.tryGetAll():
                if message.calibration != last_calibration:
                    last_calibration = message.calibration
                    print(f"Calibration: {last_calibration.value}")


if __name__ == "__main__":
    main()

"""Sensor sources, browser server, and serialization shared by the OAK4 demos."""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import depthai as dai
import numpy as np

CLOUD_SIZE = (320, 200)
PREVIEW_STRIDE = 4


def assessment_json(assessment):
    return {
        **asdict(assessment),
        "on_plane_ratio": assessment.on_plane_ratio,
        "accepted": assessment.accepted,
    }


def provenance_json(message):
    return {
        "cloud_sequence": message.getSequenceNum(),
        "source": message.source_stream_id,
        "gravity": message.gravity_direction,
        "imu_seconds": (
            message.imu_timestamp_device.total_seconds()
            if message.imu_timestamp_device is not None
            else None
        ),
    }


def cloud_preview(cloud):
    width, height = CLOUD_SIZE
    points = np.asarray(cloud.getPoints()).reshape(height, width, 3)
    sample = points[::PREVIEW_STRIDE, ::PREVIEW_STRIDE].copy()
    sample[~np.isfinite(sample).all(axis=2)] = 0
    return {
        "sequence": cloud.getSequenceNum(),
        "width": sample.shape[1],
        "height": sample.shape[0],
        "points": np.round(sample, 1).ravel().tolist(),
    }


def serve(app, address, port, frontend, shared_frontend):
    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            # OAK App frontend ports can be reused by another app.
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def json_response(self, data, status=HTTPStatus.OK):
            content = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/api/state":
                self.json_response(app.snapshot())
            else:
                if shared_frontend is not None and self.path in (
                    "/style.css",
                    "/viewer.js",
                ):
                    self.directory = str(shared_frontend)
                super().do_GET()

        def do_POST(self):
            if self.path != "/api/action":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if (
                    self.headers.get_content_type() != "application/json"
                    or not 0 < size <= 4096
                ):
                    raise ValueError("Expected a JSON action.")
                command = json.loads(self.rfile.read(size))
                if not isinstance(command, dict):
                    raise ValueError("Expected a JSON object.")
                app.command(command)
            except (ValueError, OSError) as exc:
                self.json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.json_response({"ok": True})

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer(
        (address, port), partial(Handler, directory=str(frontend))
    )


def camera_sources(pipeline, fps):
    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color = camera.requestOutput(
        CLOUD_SIZE, type=dai.ImgFrame.Type.RGB888i, fps=fps, enableUndistortion=True
    )
    depth = pipeline.create(dai.node.Depth)
    depth.build(dai.node.Depth.Algorithm.AUTO, size=CLOUD_SIZE, fps=fps)
    depth.setAlignTo(color)
    rgbd = pipeline.create(dai.node.RGBD).build()
    color.link(rgbd.inColor)
    depth.depth.link(rgbd.inDepth)
    imu = pipeline.create(dai.node.IMU)
    # The processed accelerometer uses Luxonis right/down/forward axes.
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_UNCALIBRATED, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    return rgbd.pcl, imu.out


class DemoSensors(dai.node.ThreadedHostNode):
    """Supply two planar surfaces for browser review without a camera."""

    def __init__(self):
        super().__init__()
        self.cloud = self.createOutput("cloud")
        self.imu = self.createOutput("imu")
        self.depth_mm = None

    def run(self):
        width, height = CLOUD_SIZE
        x, y = np.meshgrid(
            np.linspace(-700, 700, width), np.linspace(-400, 400, height)
        )
        z = np.where(x < 80, 800, 1100)
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3).astype(np.float32)
        shifted = points.copy()
        sequence = 0
        while self.isRunning():
            depth = self.depth_mm
            if depth is not None:
                shifted[:, 2] = depth
            cloud = dai.PointCloudData().setSize(width, height)
            cloud.setPoints(points if depth is None else shifted)
            cloud.setSequenceNum(sequence)
            cloud.setTimestamp(dai.Clock.now())
            packet = dai.IMUPacket()
            packet.acceleroMeter.z = -9.8
            imu = dai.IMUData()
            imu.packets = [packet]
            self.imu.send(imu)
            self.cloud.send(cloud)
            sequence += 1
            time.sleep(0.1)


@contextmanager
def sources(simulate, fps):
    if simulate:
        with dai.Pipeline(False) as pipeline:
            node = pipeline.create(DemoSensors)
            yield pipeline, node.cloud, node.imu, "synthetic", "synthetic-rdf-mm-v1"
        return
    with dai.Device() as device:
        if device.getPlatform() != dai.Platform.RVC4 or not device.getConnectedIMU():
            raise RuntimeError("Use an OAK4 with depth cameras and an accelerometer.")
        with dai.Pipeline(device) as pipeline:
            cloud, imu = camera_sources(pipeline, fps)
            yield (
                pipeline,
                cloud,
                imu,
                device.getDeviceId(),
                "cam-a-rgbd-rdf-mm-320x200-v1",
            )


def run_viewer(
    app, pipeline, frontend, address, port, stopping, logger, shared_frontend=None
):
    with serve(app, address, port, frontend, shared_frontend) as server:
        pipeline.start()
        Thread(target=server.serve_forever, daemon=True).start()
        logger.info("Viewer listening on port %s", server.server_port)
        try:
            while not stopping.wait(0.025) and pipeline.isRunning():
                pipeline.processTasks()
                app.update()
        finally:
            server.shutdown()

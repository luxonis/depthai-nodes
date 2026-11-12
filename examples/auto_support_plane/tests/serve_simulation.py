"""Optional offline MRE: real node/geometry/transport with synthetic camera inputs.

This is a test harness, not the OAK app entrypoint. Only the shared base class's
default-device lookup is bypassed while constructing host-only nodes.
"""

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import depthai as dai
import numpy as np

from depthai_nodes.node.base_threaded_host_node import BaseThreadedHostNode

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from controller import CLOUD_SIZE, Controller, default_settings  # noqa: E402
from frontend_server import start_frontend  # noqa: E402
from main import attach_auto_node, drain_telemetry  # noqa: E402


class Scene:
    def __init__(self):
        self.moved = False
        self.occluded = False
        self.imu = True

    def control(self, request):
        op = request.get("op")
        if op == "move":
            self.moved = not self.moved
        elif op == "occlude":
            self.occluded = not self.occluded
        elif op == "imu":
            self.imu = not self.imu
        else:
            return {"ok": False, "error": "Unknown synthetic scene action"}
        return {
            "ok": True,
            "moved": self.moved,
            "occluded": self.occluded,
            "imu": self.imu,
        }


class Source(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.cloud = self.createOutput("cloud")
        self.imu = self.createOutput("imu")
        self.scene = Scene()

    def run(self):
        width, height = CLOUD_SIZE
        x, y = np.meshgrid(
            (np.arange(width) - width / 2) * 8, (np.arange(height) - height / 2) * 8
        )
        points = np.column_stack(
            [x.ravel(), y.ravel(), np.zeros(width * height)]
        ).astype(np.float32)
        sequence = 0
        try:
            while self.isRunning():
                sequence += 1
                stamp = timedelta(seconds=time.monotonic())
                if self.scene.imu:
                    imu = dai.IMUData()
                    packet = dai.IMUPacket()
                    packet.acceleroMeter.z = -9.8
                    imu.packets = [packet]
                    imu.setTimestamp(stamp)
                    self.imu.send(imu)
                depths = np.where(x.ravel() < 400, 1000, 1550).astype(np.float32)
                if self.scene.moved:
                    depths += 300
                if self.scene.occluded:
                    depths[:] = 0
                points[:, 2] = depths
                cloud = dai.PointCloudData()
                cloud.setPoints(points)
                cloud.setSize(width, height)
                cloud.setSequenceNum(sequence)
                cloud.setTimestamp(stamp)
                cloud.setTimestampDevice(stamp)
                self.cloud.send(cloud)
                time.sleep(0.125)
        except dai.MessageQueue.QueueException:
            pass


def construct(controller, settings, initial, scene):
    pipeline = dai.Pipeline(False)
    source = pipeline.create(Source)
    source.scene = scene
    # The state machine, queues, fitting and services stay real. Production
    # main.py always gets the platform from its actual OAK4 device instead.
    with patch.object(
        BaseThreadedHostNode,
        "__init__",
        lambda self: dai.node.ThreadedHostNode.__init__(self),
    ):
        _, queues = attach_auto_node(
            pipeline, source.cloud, source.imu, controller, settings, initial
        )
    return pipeline, queues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--storage", type=Path, default=APP / "storage" / "simulation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    controller = Controller(args.storage / "support_plane.json")
    scene = Scene()
    remote = dai.RemoteConnection(
        address="127.0.0.1", webSocketPort=args.ws_port, serveFrontend=False
    )
    remote.registerService("plane_control", controller.control)
    remote.registerService("plane_demo", scene.control)
    remote.registerBinaryService("plane_cloud", controller.cloud_bytes)
    server = start_frontend(
        APP / "frontend", "127.0.0.1", args.port, args.ws_port, simulation=True
    )
    stopping = threading.Event()

    def stop(*_args):
        controller.close()
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"SYNTHETIC SCENE — http://127.0.0.1:{args.port}", flush=True)
    settings, initial = default_settings(), None
    deadline = time.monotonic() + args.seconds if args.seconds else float("inf")
    try:
        while not stopping.is_set() and time.monotonic() < deadline:
            pipeline, queues = construct(controller, settings, initial, scene)
            pipeline.start()
            request = None
            try:
                while (
                    pipeline.isRunning()
                    and not stopping.is_set()
                    and time.monotonic() < deadline
                ):
                    pipeline.processTasks()
                    drain_telemetry(queues, controller)
                    request = controller.take_restart()
                    if request is not None:
                        break
                    time.sleep(0.01)
            finally:
                with controller.condition:
                    controller.epoch += 1
                    controller.condition.notify_all()
                pipeline.stop()
                pipeline.wait()
            if request is None:
                break
            settings, initial = request
    finally:
        controller.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

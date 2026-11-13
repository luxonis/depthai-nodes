"""Commission a support plane on an OAK4 from a local browser."""

import argparse
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock

from demo_common import (
    CLOUD_SIZE,
    assessment_json,
    cloud_preview,
    provenance_json,
    run_viewer,
    sources,
)

from depthai_nodes import (
    ActiveSupportPlane,
    SupportPlaneStore,
    resolve_support_plane_path,
)
from depthai_nodes.message import (
    SupportPlaneCalibrationControl,
    SupportPlaneVerificationRequest,
)
from depthai_nodes.node import SupportPlaneCalibration, SupportPlaneCalibrationConfig

APP_ID = "com.example.support-plane-calibration"
FRONTEND = Path(__file__).resolve().parent / "frontend"
LOG = logging.getLogger(__name__)


class Commissioning:
    """Own the search controls, reviewer selection, and approval policy."""

    def __init__(self, node, cloud, store, device_id, frame_id, config):
        self.controls = node.inputControl.createInputQueue()
        self.requests = node.inputVerification.createInputQueue()
        self.batches = node.candidatesOutput.createOutputQueue(blocking=True)
        self.replies = node.verificationOutput.createOutputQueue(blocking=True)
        self.clouds = cloud.createOutputQueue(maxSize=1, blocking=False)
        self.store = store
        self.identity = {"device_id": device_id, "frame_id": frame_id}
        self.distance = config.check_distance_mm
        self.lock = Lock()
        self.searching = False
        self.attempt = 0
        self.batch = None
        self.preview = None
        self.pending = None
        self.verification = None
        self.saved = None
        self.active = None
        self.error = None
        try:
            self.saved = store.load(**self.identity)
        except (OSError, ValueError) as exc:
            self.error = str(exc)

    def _request(self, plane, candidate):
        if self.pending is not None:
            raise ValueError("A verification is already pending.")
        request_id = f"check-{time.monotonic_ns()}"
        if not self.requests.trySend(
            SupportPlaneVerificationRequest(plane, request_id)
        ):
            raise ValueError("The verification queue is busy. Try again.")
        self.pending = request_id, candidate
        self.verification = None

    def _control(self, message):
        if not self.controls.trySend(message):
            raise ValueError("The control queue is busy. Try again.")

    def command(self, command):
        with self.lock:
            action = command.get("action")
            if action in ("start", "stop"):
                start = action == "start"
                if start == self.searching:
                    return
                self._control(
                    SupportPlaneCalibrationControl.start()
                    if start
                    else SupportPlaneCalibrationControl.stop()
                )
                self.searching = start
                if start:
                    self.batch = None
                    self.verification = None
            elif action == "verify":
                index = command.get("index")
                if (
                    self.searching
                    or self.batch is None
                    or command.get("attempt") != self.attempt
                    or not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < len(self.batch.assessments)
                ):
                    raise ValueError("Select a candidate from the current fit.")
                self._request(
                    self.batch.assessments[index].plane, (self.attempt, index)
                )
            elif action == "restore":
                if self.saved is None:
                    raise ValueError("There is no compatible saved approval.")
                self._request(self.saved.plane, None)
            elif action == "approve":
                if self.verification is None:
                    raise ValueError("Verify the selected candidate before approval.")
                reply, candidate = self.verification
                if (
                    candidate is None
                    or not reply.assessment.accepted
                    or candidate != (self.attempt, command.get("index"))
                    or command.get("request_id") != reply.request_id
                    or self.searching
                ):
                    raise ValueError(
                        "Approval requires a passing check for this candidate."
                    )
                record = ActiveSupportPlane(
                    plane=reply.assessment.plane,
                    approved_at=datetime.now(timezone.utc),
                    **self.identity,
                )
                self.store.save(record)
                self.saved = self.active = record
                self.error = None
            elif action == "invalidate":
                self.store.invalidate()
                self.saved = self.active = self.pending = self.verification = None
                self.error = None
            else:
                raise ValueError("Unknown action.")

    def update(self):
        with self.lock:
            cloud = self.clouds.tryGet()
            if cloud is not None:
                self.preview = cloud_preview(cloud)
            for batch in self.batches.tryGetAll():
                self.attempt += 1
                self.batch = batch
                if any(item.accepted for item in batch.assessments):
                    self.searching = False
            for reply in self.replies.tryGetAll():
                if self.pending is None or reply.request_id != self.pending[0]:
                    continue
                _, candidate = self.pending
                self.pending = None
                self.verification = reply, candidate
                if candidate is None and reply.assessment.accepted:
                    self.active = self.saved

    def snapshot(self):
        with self.lock:
            batch = self.batch
            reply, candidate = self.verification or (None, None)
            return {
                "searching": self.searching,
                "attempt": self.attempt,
                "batch": (
                    {
                        **provenance_json(batch),
                        "assessments": [assessment_json(a) for a in batch.assessments],
                    }
                    if batch is not None
                    else None
                ),
                "verification": (
                    {
                        **provenance_json(reply),
                        **assessment_json(reply.assessment),
                        "request_id": reply.request_id,
                        "candidate": candidate,
                    }
                    if reply is not None
                    else None
                ),
                "pending": self.pending is not None,
                "active": self.active.plane.toJson() if self.active else None,
                "saved_at": self.saved.approved_at.isoformat() if self.saved else None,
                "distance": self.distance,
                "preview": self.preview,
                "error": self.error,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument(
        "--height-range", nargs=2, type=float, metavar=("MIN_MM", "MAX_MM")
    )
    parser.add_argument("--installation", default=os.getenv("INSTALLATION_ID", "bench"))
    parser.add_argument("--store", type=Path)
    args = parser.parse_args()
    if not 0 < args.fps <= 30:
        parser.error("--fps must be greater than 0 and at most 30")
    logging.basicConfig(level=logging.INFO)
    stopping = Event()

    def stop(*_args):
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        config = SupportPlaneCalibrationConfig(
            vertical_distance_range_mm=tuple(args.height_range)
            if args.height_range
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    app_id = APP_ID + ".simulation" if args.simulate else APP_ID
    store = SupportPlaneStore(
        resolve_support_plane_path(app_id, args.installation, args.store)
    )
    with sources(args.simulate, args.fps) as (
        pipeline,
        cloud,
        imu,
        device_id,
        frame_id,
    ):
        node = pipeline.create(SupportPlaneCalibration).build(
            cloud, imu, CLOUD_SIZE, config
        )
        app = Commissioning(
            node, cloud, store, device_id, f"{frame_id}/{args.installation}", config
        )
        address = "127.0.0.1" if args.simulate else "0.0.0.0"
        run_viewer(app, pipeline, FRONTEND, address, args.port, stopping, LOG)


if __name__ == "__main__":
    main()

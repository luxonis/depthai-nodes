"""Show automatic support plane selection and refitting on an OAK4."""

import argparse
import logging
import signal
import sys
from pathlib import Path
from threading import Event, Lock

from depthai_nodes.message import SupportPlane
from depthai_nodes.node import (
    AutoSupportPlane,
    AutoSupportPlaneConfig,
    SupportPlaneCalibrationConfig,
)

DIRECTORY = Path(__file__).resolve().parent
SHARED = DIRECTORY.parent / "support_plane_calibration"
# A staged app includes the shared module and assets in its own directory.
if not (DIRECTORY / "demo_common.py").is_file():
    sys.path.insert(0, str(SHARED))

from demo_common import (  # noqa: E402
    CLOUD_SIZE,
    assessment_json,
    cloud_preview,
    provenance_json,
    run_viewer,
    sources,
)

FRONTEND = DIRECTORY / "frontend"
LOG = logging.getLogger(__name__)


class AutomaticViewer:
    """Read the node's three outputs and expose the latest evidence to the UI."""

    def __init__(self, node, cloud, config, mode, demo):
        self.results = node.out.createOutputQueue(blocking=True)
        self.checks = node.verificationOutput.createOutputQueue(blocking=True)
        self.batches = node.candidatesOutput.createOutputQueue(blocking=True)
        self.clouds = cloud.createOutputQueue(maxSize=1, blocking=False)
        self.demo = demo
        self.lock = Lock()
        self.state = {
            "preview": None,
            "result": None,
            "check": None,
            "batch": None,
            "mode": mode.value,
            "distance": config.check_distance_mm,
            "synthetic": demo is not None,
        }

    def update(self):
        with self.lock:
            cloud = self.clouds.tryGet()
            if cloud is not None:
                self.state["preview"] = cloud_preview(cloud)
            for batch in self.batches.tryGetAll():
                self.state["batch"] = {
                    **provenance_json(batch),
                    "assessments": [assessment_json(a) for a in batch.assessments],
                }
            for check in self.checks.tryGetAll():
                self.state["check"] = {
                    **provenance_json(check),
                    **assessment_json(check.assessment),
                    "request_id": check.request_id,
                }
            for result in self.results.tryGetAll():
                self.state["result"] = {
                    "passed": result.passed,
                    "plane": result.plane.toJson()
                    if result.plane is not None
                    else None,
                    "cloud_sequence": result.getSequenceNum(),
                }

    def command(self, command):
        depth = command.get("depth_mm")
        if (
            self.demo is None
            or command.get("action") != "scene"
            or depth not in (800, 1100)
        ):
            raise ValueError("Scene changes are available in synthetic mode.")
        self.demo.depth_mm = depth

    def snapshot(self):
        with self.lock:
            return self.state.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in AutoSupportPlaneConfig.Mode],
        default=AutoSupportPlaneConfig.Mode.CONTINUOUS.value,
    )
    parser.add_argument(
        "--initial-plane", type=Path, help="Verify this plane before use"
    )
    args = parser.parse_args()
    initial = (
        SupportPlane.fromJsonFile(args.initial_plane) if args.initial_plane else None
    )
    mode = AutoSupportPlaneConfig.Mode(args.mode)
    config = AutoSupportPlaneConfig(
        mode=mode,
        calibration=SupportPlaneCalibrationConfig(),
        check_interval_sec=0.5,
        failed_checks_before_refit=3,
    )

    logging.basicConfig(level=logging.INFO)
    stopping = Event()

    def stop(*_args):
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with sources(args.simulate, 8) as (pipeline, cloud, imu, _, _):
        node = pipeline.create(AutoSupportPlane).build(
            cloud, imu, CLOUD_SIZE, config, initialPlane=initial
        )
        demo = cloud.getParent() if args.simulate else None
        app = AutomaticViewer(node, cloud, config.calibration, mode, demo)
        address = "127.0.0.1" if args.simulate else "0.0.0.0"
        run_viewer(
            app,
            pipeline,
            FRONTEND,
            address,
            args.port,
            stopping,
            LOG,
            shared_frontend=SHARED / "frontend" if SHARED.is_dir() else None,
        )


if __name__ == "__main__":
    main()

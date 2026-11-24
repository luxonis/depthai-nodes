#!/usr/bin/env python3
"""Record labeled OAK IMU data for camera movement detection."""

import argparse
import json
import math
import shutil
import tarfile
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import depthai as dai

# Cues label the intended action window. The camera can settle anywhere in
# that window. The following hold phase labels the resulting pose.
DESK_TEST = (
    ("baseline", 15, "Leave the camera still. Marking the calibration request."),
    ("slow_tilt", 12, "Slowly tilt the camera down by about 10 degrees."),
    ("hold_tilt", 8, "Hold the new tilt. Rest it on a wedge if possible."),
    ("return_from_tilt", 6, "Return to the original pose, then let go."),
    ("settle_after_tilt", 6, "Leave the camera still."),
    ("quick_tilt", 3, "Quickly tilt it down about 10 degrees. Hold it there."),
    ("hold_quick_tilt", 8, "Keep the camera at the new tilt."),
    ("return_from_quick_tilt", 6, "Return to the original pose, then let go."),
    ("settle_after_quick_tilt", 6, "Leave the camera still."),
    ("slow_yaw", 5, "Keep it flat and turn about 20 degrees over 4–5 seconds."),
    ("hold_yaw", 8, "Keep it at the new heading."),
    ("return_from_yaw", 6, "Turn back to the original heading, then let go."),
    ("settle_after_yaw", 6, "Leave the camera still."),
    ("quick_yaw", 3, "Keep it flat and quickly turn it about 20 degrees."),
    ("hold_quick_yaw", 8, "Keep it at the new heading."),
    ("return_from_quick_yaw", 6, "Turn back to the original heading, then let go."),
    ("settle_after_quick_yaw", 6, "Leave the camera still."),
    ("bump", 6, "Tap the desk near the camera a few times without shifting it."),
    ("settle_after_bump", 6, "Leave the camera still."),
    ("magnetic_disturbance", 10, "Move a steel tool near the camera and back away."),
    ("final_hold", 10, "Leave the camera still."),
)
INERTIAL_REPORTS = (
    ("acceleroMeter", "ACCELEROMETER_RAW", 100),
    ("gyroscope", "GYROSCOPE_RAW", 100),
)
MAGNETIC_REPORTS = (("magneticField", "MAGNETOMETER_RAW", 50),)


def write_row(stream, **row):
    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def sample_reports(message, reports, seen, counts, stream):
    for packet in message.packets:
        for field, sensor, _ in reports:
            report = getattr(packet, field)
            device_time = report.getTimestampDevice().total_seconds()
            host_time = report.getTimestamp().total_seconds()
            xyz = (report.x, report.y, report.z)
            # Unpopulated report fields have a zero device timestamp. Some
            # packet layouts also repeat the last sample from a slower sensor.
            if device_time <= 0 or not all(map(math.isfinite, xyz)):
                continue
            key = (report.getSequenceNum(), device_time)
            if seen.get(field) == key:
                continue
            seen[field] = key
            counts[sensor] += 1
            write_row(
                stream,
                type="sample",
                sensor=sensor,
                device_time_s=device_time,
                host_time_s=host_time,
                sequence=key[0],
                accuracy=int(report.accuracy),
                xyz=xyz,
            )


def capture(pipeline, directory, phases, reports):
    config = dai.RecordConfig()
    config.outputDir = str(directory.resolve())
    config.syncCameraOutputs = False
    pipeline.enableHolisticRecord(config)

    imu = pipeline.create(dai.node.IMU)
    for _, sensor, hz in reports:
        imu.enableIMUSensor(getattr(dai.IMUSensor, sensor), hz)
    imu.setBatchReportThreshold(4)
    imu.setMaxBatchReports(20)
    queue = imu.out.createOutputQueue(maxSize=256, blocking=False)
    device = pipeline.getDefaultDevice()
    counts = Counter()
    seen = {}

    with (directory / "imu.jsonl").open("x", encoding="utf-8", buffering=65536) as log:
        write_row(
            log,
            type="metadata",
            depthai=dai.__version__,
            device=str(device.getProductName()),
            platform=str(device.getPlatform()),
            imu=device.getConnectedIMU(),
            requested_rates_hz={sensor: hz for _, sensor, hz in reports},
        )
        pipeline.start()
        completed = False
        try:
            # Drain the startup samples before cueing the first action.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not all(
                counts[sensor] for _, sensor, _ in reports
            ):
                message = queue.get(timedelta(milliseconds=100))
                if message is not None:
                    sample_reports(message, reports, seen, counts, log)
            missing = [sensor for _, sensor, _ in reports if not counts[sensor]]
            if missing:
                raise RuntimeError(
                    f"No usable reports received for {', '.join(missing)}."
                )

            for phase, duration, instruction in phases:
                write_row(
                    log,
                    type="phase",
                    phase=phase,
                    host_time_s=dai.Clock.now().total_seconds(),
                    duration_s=duration,
                    calibrate=phase == "baseline",
                )
                print(f"{phase} ({duration:g}s) — {instruction}", flush=True)
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline:
                    message = queue.get(timedelta(milliseconds=25))
                    if message is not None:
                        sample_reports(message, reports, seen, counts, log)
                    if not pipeline.isRunning():
                        raise RuntimeError("The IMU pipeline stopped during capture.")
            completed = True
        except KeyboardInterrupt:
            print("Interrupted. Saving the partial capture.", flush=True)
        finally:
            write_row(
                log,
                type="end",
                host_time_s=dai.Clock.now().total_seconds(),
                completed=completed,
            )
            pipeline.stop()
    return counts


def record(args):
    directory = args.out
    if directory.exists():
        raise FileExistsError(
            f"Choose a new capture directory. Already exists: {directory}"
        )
    reports = MAGNETIC_REPORTS if args.magnetometer_only else INERTIAL_REPORTS
    phases = DESK_TEST
    if args.stationary_seconds is not None:
        phases = (("baseline", args.stationary_seconds, "Leave the camera still."),)
    print(f"Recording to {directory.resolve()}", flush=True)
    with dai.Pipeline(
        dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    ) as pipeline:
        directory.mkdir(parents=True, exist_ok=False)
        counts = capture(pipeline, directory, phases, reports)
    # The SDK creates the archive when the pipeline object is destroyed.
    del pipeline
    print("Samples:", dict(counts), flush=True)
    archive = next(directory.glob("recording_*.tar"), None)
    if archive is None:
        raise RuntimeError(
            f"DepthAI did not write an archive. See {directory / 'imu.jsonl'}."
        )
    with tarfile.open(archive, "r:*") as tar:
        if not tar.getmember("IMU.mcap").size:
            raise RuntimeError(f"IMU recording is empty in {archive}.")
    print(f"Archive: {archive.resolve()}")
    print(f"Labels and sensor readings: {(directory / 'imu.jsonl').resolve()}")


def replay(archive, seconds):
    """Read the native IMU messages on the host without opening a camera."""
    with TemporaryDirectory() as temporary:
        imu_file = Path(temporary) / "IMU.mcap"
        with tarfile.open(archive, "r:*") as tar:
            with tar.extractfile("IMU.mcap") as source, imu_file.open("wb") as target:
                shutil.copyfileobj(source, target)
        with dai.Pipeline(False) as pipeline:
            replay_node = pipeline.create(dai.node.ReplayMetadataOnly)
            replay_node.setReplayFile(imu_file).setLoop(True)
            queue = replay_node.out.createOutputQueue(maxSize=256, blocking=False)
            pipeline.start()
            deadline = time.monotonic() + seconds
            packets = 0
            while time.monotonic() < deadline:
                message = queue.get(timedelta(milliseconds=100))
                if isinstance(message, dai.IMUData) and message.packets:
                    if not packets:
                        packet = message.packets[0]
                        for field, sensor, _ in INERTIAL_REPORTS + MAGNETIC_REPORTS:
                            report = getattr(packet, field)
                            if report.getTimestampDevice() > timedelta(0):
                                print(
                                    f"First {sensor}:", (report.x, report.y, report.z)
                                )
                    packets += len(message.packets)
            pipeline.stop()
    if not packets:
        raise RuntimeError("No IMU packets were replayed.")
    print(f"Replayed {packets} IMU packets from {archive}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="New directory for a capture")
    mode.add_argument("--replay", type=Path, help="Inspect a saved IMU tar on the host")
    parser.add_argument("--device", help="Device IP address or ID for capture")
    sensors = parser.add_mutually_exclusive_group()
    sensors.add_argument(
        "--magnetometer-only", action="store_true", help="Capture only magnetic field"
    )
    sensors.add_argument(
        "--no-magnetometer", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--stationary-seconds", type=float, help="Capture only a still hold"
    )
    parser.add_argument("--replay-seconds", type=float, default=3)
    args = parser.parse_args()
    for value in (args.stationary_seconds, args.replay_seconds):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("durations must be positive and finite")
    if args.replay:
        replay(args.replay, args.replay_seconds)
    else:
        record(args)


if __name__ == "__main__":
    main()

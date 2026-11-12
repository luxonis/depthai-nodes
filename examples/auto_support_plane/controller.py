"""Thread-safe UI bridge.

AutoSupportPlane owns the actual calibration protocol.
"""

import copy
import json
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from depthai_nodes.message import SupportPlane
from depthai_nodes.node import AutoSupportPlaneConfig, PlaneCalibrationConfig

CLOUD_SIZE = (320, 200)


def default_settings():
    return {
        "mode": "continuous",
        "selection": "manual",
        "check_interval_sec": 0.5,
        "failed_checks_before_refit": 3,
        "calibration": asdict(PlaneCalibrationConfig(max_calibration_attempts=10)),
    }


def make_config(settings):
    if set(settings) != set(default_settings()):
        raise ValueError("Settings must contain exactly the documented fields")
    if settings["selection"] not in ("manual", "automatic"):
        raise ValueError("Selection must be manual or automatic")
    if set(settings["calibration"]) != set(default_settings()["calibration"]):
        raise ValueError("Calibration must contain exactly the documented fields")
    return AutoSupportPlaneConfig(
        mode=AutoSupportPlaneConfig.Mode(settings["mode"]),
        check_interval_sec=settings["check_interval_sec"],
        failed_checks_before_refit=settings["failed_checks_before_refit"],
        calibration=PlaneCalibrationConfig(**settings["calibration"]),
    )


def plane_json(plane):
    return None if plane is None else plane.toJson()


class Controller:
    def __init__(self, saved_path: Path):
        self.saved_path = saved_path
        self.condition = threading.Condition(threading.RLock())
        self.epoch = 0
        self.ticket = 0
        self.pending = None
        self.answer = None
        self.closed = False
        self.restart = None
        self.frames = OrderedDict()
        self.latest_frame = None
        self.fit_candidates = None
        self.fit_sequence = None
        self.candidate_sequence = None
        self.held = None
        self.last_saved_plane = None
        self.preview_index = 0
        self.cloud_at = 0.0
        self.imu_at = 0.0
        self.events = deque(maxlen=30)
        self.data = {
            "phase": "starting",
            "settings": default_settings(),
            "candidates": [],
            "last_result": None,
            "verification": None,
            "calibration_outcome": None,
            "error": None,
        }

    def event(self, text):
        self.events.append({"at": time.strftime("%H:%M:%S"), "text": text})

    def begin(self, settings, initial_plane=None):
        make_config(settings)
        with self.condition:
            self.epoch += 1
            self.pending = self.answer = None
            self.frames.clear()
            self.latest_frame = None
            self.fit_candidates = self.fit_sequence = self.candidate_sequence = None
            self.held = initial_plane
            self.cloud_at = self.imu_at = 0.0
            self.data.update(
                phase="verifying restored plane" if initial_plane else "searching",
                settings=copy.deepcopy(settings),
                candidates=[],
                last_result=None,
                verification=None,
                calibration_outcome=None,
                error=None,
            )
            self.event(f"Started {settings['mode']} / {settings['selection']}")
            self.condition.notify_all()
            return self.epoch

    def select_candidate(self, candidates, epoch):
        """Run on the node thread; returning None demonstrates rejecting the fit."""
        with self.condition:
            if epoch != self.epoch or self.closed or self.restart is not None:
                return None
            self.ticket += 1
            self.pending = candidates
            self.answer = None
            self.preview_index = 0
            self.candidate_sequence = (
                self.fit_sequence if self.fit_candidates is candidates else None
            )
            self.data["phase"] = "choose a plane"
            self.data["candidates"] = [
                {
                    "plane": plane_json(c.plane),
                    "inlier_count": c.inlier_count,
                    "inlier_ratio": c.inlier_ratio,
                }
                for c in candidates
            ]
            self.event(
                f"Found {len(candidates)} candidate(s); waiting for confirmation"
            )
            self.condition.wait_for(
                lambda: self.answer is not None
                or self.closed
                or self.restart is not None
                or self.epoch != epoch
            )
            if self.closed or self.restart is not None or self.epoch != epoch:
                return None
            action, index = self.answer
            selected = candidates[index] if action == "confirm" else None
            self.pending = self.answer = None
            self.candidate_sequence = None
            self.data["phase"] = "confirming" if selected else "searching"
            self.event(
                "Candidate confirmed" if selected else "Rejected all; trying again"
            )
            return selected

    def record_calibration(self, result):
        with self.condition:
            self.data["calibration_outcome"] = result.outcome.value
            self.fit_candidates = result.candidates
            self.fit_sequence = result.getSequenceNum()
            # These are host-only Python Buffer messages: both consumers receive
            # the same immutable candidates tuple. Never associate by geometry:
            # two consecutive fits may return identical equations.
            if self.pending is result.candidates:
                self.candidate_sequence = self.fit_sequence
            if result.candidates:
                if self.data["settings"]["selection"] == "automatic":
                    self.data["candidates"] = [
                        {
                            "plane": plane_json(c.plane),
                            "inlier_count": c.inlier_count,
                            "inlier_ratio": c.inlier_ratio,
                        }
                        for c in result.candidates
                    ]
            else:
                self.event(f"Calibration {result.outcome.value}")

    def record_verification(self, result):
        with self.condition:
            self.data["verification"] = {
                "accepted": result.accepted,
                "gravity_angle_deg": result.gravity_angle_deg,
                "on_plane_ratio": result.on_plane_ratio,
                "sequence": result.getSequenceNum(),
            }
            if not result.accepted:
                self.event("Plane check failed: coverage, tilt, or insufficient depth")

    def record_result(self, result):
        with self.condition:
            self.held = result.plane
            self.data["last_result"] = {
                "passed": result.passed,
                "plane": plane_json(result.plane),
                "sequence": result.getSequenceNum(),
            }
            if self.pending is None:
                self.data["phase"] = (
                    "holding" if result.plane is not None else "searching"
                )
                if (
                    result.plane is None
                    and self.data["settings"]["mode"] == "on_start"
                    and self.data["calibration_outcome"] == "failed"
                ):
                    self.data["phase"] = "attempt failed — restart to try again"
            if result.passed and result.plane is not None:
                # Save only a newly confirmed plane, not every passing check.
                if (
                    self.last_saved_plane != result.plane
                    or not self.saved_path.exists()
                ):
                    self.saved_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.saved_path.with_suffix(".tmp")
                    result.plane.toJsonFile(temporary)
                    temporary.replace(self.saved_path)
                    self.last_saved_plane = result.plane
                self.event("Plane found / verified")
            elif result.plane is not None:
                self.event("Refit failed; retaining the previous plane")
            else:
                self.event("No plane in use; searching")

    def record_cloud(self, cloud):
        points = np.asarray(cloud.getPoints())
        width, height = CLOUD_SIZE
        if points.shape != (width * height, 3):
            raise ValueError(
                f"Expected organized {CLOUD_SIZE} cloud, got {points.shape}"
            )
        # Display 1/4 of the points. Calibration always receives the full cloud.
        sampled = points.reshape(height, width, 3)[::2, ::2].reshape(-1, 3)
        sampled = sampled[np.isfinite(sampled).all(axis=1) & (sampled[:, 2] > 0)]
        packed = sampled.astype("<f4").tobytes()
        sequence = cloud.getSequenceNum()
        with self.condition:
            self.frames[sequence] = (len(sampled), packed)
            self.latest_frame = sequence
            self.cloud_at = time.monotonic()
            while len(self.frames) > 32:
                # Keep the candidate's exact source frame throughout selection.
                oldest = next(iter(self.frames))
                if oldest == self.candidate_sequence:
                    self.frames.move_to_end(oldest)
                else:
                    del self.frames[oldest]

    def record_imu(self, message):
        if any(
            np.linalg.norm([p.acceleroMeter.x, p.acceleroMeter.y, p.acceleroMeter.z])
            > 1e-6
            for p in message.packets
        ):
            with self.condition:
                self.imu_at = time.monotonic()

    def snapshot(self):
        with self.condition:
            data = copy.deepcopy(self.data)
            now = time.monotonic()
            data.update(
                epoch=self.epoch,
                ticket=self.ticket,
                waiting=self.pending is not None,
                preview_index=self.preview_index,
                held_plane=plane_json(self.held),
                saved=self.saved_path.exists(),
                cloud_age_sec=now - self.cloud_at if self.cloud_at else None,
                imu_age_sec=now - self.imu_at if self.imu_at else None,
                events=list(self.events),
            )
            return data

    def cloud_bytes(self, _request=b""):
        """Binary service: uint32 JSON length, JSON metadata, float32 XYZ in mm."""
        with self.condition:
            frozen = self.pending is not None
            sequence = self.candidate_sequence if frozen else self.latest_frame
            count, packed = self.frames.get(sequence, (0, b""))
            plane = self.pending[self.preview_index].plane if frozen else self.held
            metadata = json.dumps(
                {
                    "sequence": sequence,
                    "count": count,
                    "frozen": frozen,
                    "epoch": self.epoch,
                    "ticket": self.ticket,
                    "plane": plane_json(plane),
                    "band_mm": self.data["settings"]["calibration"][
                        "check_distance_mm"
                    ],
                },
                separators=(",", ":"),
            ).encode()
        return struct.pack("<I", len(metadata)) + metadata + packed

    def control(self, request):
        """All service errors become responses, never escape into the SDK thread."""
        try:
            with self.condition:
                op = request["op"]
                if op in ("preview", "confirm", "reject"):
                    if (
                        request.get("epoch") != self.epoch
                        or request.get("ticket") != self.ticket
                        or self.pending is None
                        or self.answer is not None
                        or self.restart is not None
                    ):
                        raise ValueError(
                            "This selection expired; refresh the candidates"
                        )
                    index = request.get("index", 0)
                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or not 0 <= index < len(self.pending)
                    ):
                        raise ValueError("Invalid candidate index")
                    if op == "preview":
                        self.preview_index = index
                    else:
                        self.answer = (
                            "confirm" if op == "confirm" else "reject",
                            index,
                        )
                        self.condition.notify_all()
                elif op in ("restart", "restore", "forget"):
                    settings = request.get("settings", self.data["settings"])
                    make_config(settings)
                    plane = None
                    if op == "restore":
                        plane = SupportPlane.fromJson(request["plane"])
                    if op == "forget":
                        self.saved_path.unlink(missing_ok=True)
                    self.restart = (copy.deepcopy(settings), plane)
                    self.data["phase"] = "restarting"
                    self.condition.notify_all()
                elif op == "export":
                    if (
                        self.held is None
                        or self.data["phase"] == "verifying restored plane"
                    ):
                        raise ValueError("No confirmed plane to export")
                    return {"ok": True, "plane": plane_json(self.held)}
                elif op != "state":
                    raise ValueError(f"Unknown operation: {op}")
                return {"ok": True, "state": self.snapshot()}
        except (KeyError, TypeError, ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

    def take_restart(self):
        with self.condition:
            request, self.restart = self.restart, None
            if request is not None:
                # Invalidate before clearing the restart flag, so a selector
                # that has not awakened yet can still observe cancellation.
                self.epoch += 1
                self.condition.notify_all()
            return request

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

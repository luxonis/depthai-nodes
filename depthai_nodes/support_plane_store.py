"""Application-owned storage for approved support planes."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

from .message.support_plane import SupportPlane

_SCHEMA_VERSION = 1
_FILENAME = "support-plane.json"


@dataclass(frozen=True, kw_only=True)
class ActiveSupportPlane:
    """An explicitly approved plane for a particular device and frame.

    Attributes:
        plane: The approved equation in millimeters.
        device_id: The device's stable identity.
        frame_id: Identity of the coordinate frame and installation configuration.
        approved_at: A timezone-aware approval time, stored in UTC.
    """

    plane: SupportPlane
    device_id: str
    frame_id: str
    approved_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.plane, SupportPlane):
            raise TypeError("plane must be a SupportPlane")
        for field in ("device_id", "frame_id"):
            value = getattr(self, field)
            if not isinstance(value, str):
                raise TypeError(f"{field} must be a string")
            if not value.strip():
                raise ValueError(f"{field} must be nonempty")
        if not isinstance(self.approved_at, datetime):
            raise TypeError("approved_at must be a datetime")
        if self.approved_at.utcoffset() is None:
            raise ValueError("approved_at must be timezone-aware")
        object.__setattr__(
            self, "approved_at", self.approved_at.astimezone(timezone.utc)
        )


class SupportPlaneStore:
    """Read, save, and invalidate an application's approved plane at ``path``."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser()

    def save(self, record: ActiveSupportPlane) -> None:
        """Atomically replace the stored approval with ``record``.

        Raises:
            OSError: If the record cannot be written or replaced.
            TypeError: If ``record`` is not an `ActiveSupportPlane`.
        """
        if not isinstance(record, ActiveSupportPlane):
            raise TypeError("record must be an ActiveSupportPlane")
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "plane": record.plane.toJson(),
            "device_id": record.device_id,
            "frame_id": record.frame_id,
            "approved_at": record.approved_at.isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, allow_nan=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load(self, *, device_id: str, frame_id: str) -> ActiveSupportPlane | None:
        """Return a compatible approval or ``None`` if the file is missing.

        Raises:
            OSError: If the file exists but cannot be read.
            ValueError: If the record is invalid, unsupported, or incompatible.
        """
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"Invalid support plane record at {self.path}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid support plane record at {self.path}")
        version = payload.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("Support plane schema version must be an integer")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported support plane schema version: {version!r}")
        try:
            approved_at = payload["approved_at"]
            if not isinstance(approved_at, str):
                raise TypeError("approved_at must be an ISO 8601 timestamp string")
            if approved_at.endswith("Z"):
                approved_at = approved_at[:-1] + "+00:00"
            record = ActiveSupportPlane(
                plane=SupportPlane.fromJson(payload["plane"]),
                device_id=payload["device_id"],
                frame_id=payload["frame_id"],
                approved_at=datetime.fromisoformat(approved_at),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid support plane record at {self.path}") from exc
        if record.device_id != device_id:
            raise ValueError("Stored support plane belongs to a different device")
        if record.frame_id != frame_id:
            raise ValueError("Stored support plane belongs to a different frame")
        return record

    def invalidate(self) -> None:
        """Remove the approval if present.

        Propagate any other I/O error.
        """
        self.path.unlink(missing_ok=True)


def resolve_support_plane_path(
    app_id: str, installation_id: str, path: str | os.PathLike[str] | None = None
) -> Path:
    """Resolve an explicit path, OAK app storage, or the host per-user state path.

    Each host installation gets its own directory. An OAK app that manages multiple
    installations can supply a separate explicit path for each one.
    """
    if path is not None:
        return Path(path).expanduser()
    oak_storage = os.environ.get("OAKAGENT_STORAGE_APP")
    if oak_storage:
        return Path(oak_storage) / _FILENAME

    for label, value in (("app_id", app_id), ("installation_id", installation_id)):
        if (
            not isinstance(value, str)
            or not value.strip()
            or value in {".", ".."}
            or any(character in value for character in ("/", "\\", "\0"))
            or PureWindowsPath(value).drive
        ):
            raise ValueError(f"{label} must be a single directory name")

    if sys.platform == "darwin":
        state_dir = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        state_dir = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
    else:
        state_dir = Path(os.environ.get("XDG_STATE_HOME", ""))
        if not state_dir.is_absolute():
            state_dir = Path.home() / ".local" / "state"
    return state_dir / app_id / installation_id / _FILENAME

import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from depthai_nodes import (
    ActiveSupportPlane,
    SupportPlane,
    SupportPlaneStore,
    resolve_support_plane_path,
)

APPROVAL = ActiveSupportPlane(
    plane=SupportPlane(normal=(0, 0, 1), offset_mm=-800),
    device_id="oak-123",
    frame_id="stereo-left/conveyor-v2",
    approved_at=datetime(2026, 11, 13, 12, tzinfo=timezone.utc),
)
IDENTITY = {"device_id": APPROVAL.device_id, "frame_id": APPROVAL.frame_id}


def test_commission_restore_and_invalidate(tmp_path):
    path = tmp_path / "line-4" / "support-plane.json"
    store = SupportPlaneStore(path)

    assert store.load(**IDENTITY) is None
    store.invalidate()
    store.save(APPROVAL)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "plane": {"normal": [0.0, 0.0, 1.0], "offset_mm": -800.0},
        "device_id": "oak-123",
        "frame_id": "stereo-left/conveyor-v2",
        "approved_at": "2026-11-13T12:00:00+00:00",
    }
    restarted_store = SupportPlaneStore(path)
    assert restarted_store.load(**IDENTITY) == APPROVAL

    updated = replace(APPROVAL, plane=SupportPlane(normal=(0, 0, 1), offset_mm=-850))
    restarted_store.save(updated)
    assert store.load(**IDENTITY) == updated

    restarted_store.invalidate()
    restarted_store.invalidate()
    assert store.load(**IDENTITY) is None


@pytest.mark.parametrize("field", ["device_id", "frame_id"])
def test_restore_rejects_an_incompatible_identity(tmp_path, field):
    store = SupportPlaneStore(tmp_path / "support-plane.json")
    store.save(APPROVAL)

    with pytest.raises(ValueError):
        store.load(**(IDENTITY | {field: "different"}))
    assert store.load(**IDENTITY) == APPROVAL


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"device_id": 123},
        {"frame_id": ""},
        {"approved_at": None},
        {"approved_at": "2026-11-13T12:00:00"},
        {"plane": {"normal": [0, 0, 0], "offset_mm": 0}},
        {"plane": {"normal": [0, 0, 1], "offset_mm": 10**500}},
    ],
)
def test_restore_rejects_invalid_or_unsupported_records(tmp_path, changes):
    path = tmp_path / "support-plane.json"
    store = SupportPlaneStore(path)
    store.save(APPROVAL)
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload | changes), encoding="utf-8")

    with pytest.raises(ValueError):
        store.load(**IDENTITY)


@pytest.mark.parametrize("contents", [b"{", b"\xff", b"[]", b'{"schema_version":1}'])
def test_restore_rejects_malformed_files(tmp_path, contents):
    path = tmp_path / "support-plane.json"
    path.write_bytes(contents)

    with pytest.raises(ValueError):
        SupportPlaneStore(path).load(**IDENTITY)


def test_restore_reports_io_errors(tmp_path):
    path = tmp_path / "support-plane.json"
    path.mkdir()

    with pytest.raises(OSError):
        SupportPlaneStore(path).load(**IDENTITY)


def test_failed_replacement_keeps_previous_approval(tmp_path, monkeypatch):
    path = tmp_path / "support-plane.json"
    store = SupportPlaneStore(path)
    store.save(APPROVAL)
    updated = replace(APPROVAL, approved_at=APPROVAL.approved_at + timedelta(days=1))

    def fail_replace(source, destination):
        assert Path(source).parent == destination.parent
        assert store.load(**IDENTITY) == APPROVAL
        raise OSError("replacement failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.save(updated)

    assert store.load(**IDENTITY) == APPROVAL
    assert list(tmp_path.iterdir()) == [path]


def test_approval_time_is_normalized_to_utc():
    local_time = datetime(2026, 11, 13, 13, tzinfo=timezone(timedelta(hours=1)))
    assert replace(APPROVAL, approved_at=local_time).approved_at == APPROVAL.approved_at
    with pytest.raises(ValueError):
        replace(APPROVAL, approved_at=local_time.replace(tzinfo=None))


def test_resolve_prefers_explicit_then_oak_storage(tmp_path, monkeypatch):
    explicit = tmp_path / "approved.json"
    oak_storage = tmp_path / "oak"
    monkeypatch.setenv("OAKAGENT_STORAGE_APP", str(oak_storage))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert resolve_support_plane_path("inspection", "line-4", str(explicit)) == explicit
    assert (
        resolve_support_plane_path("inspection", "line-4")
        == oak_storage / "support-plane.json"
    )


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_resolve_scopes_host_defaults_by_app_and_installation(
    tmp_path, monkeypatch, platform
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    monkeypatch.delenv("OAKAGENT_STORAGE_APP", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    base = {
        "linux": tmp_path / "user" / ".local" / "state",
        "darwin": tmp_path / "user" / "Library" / "Application Support",
        "win32": tmp_path / "user" / "AppData" / "Local",
    }[platform]

    assert (
        resolve_support_plane_path("inspection", "line-4")
        == base / "inspection" / "line-4" / "support-plane.json"
    )


def test_linux_resolver_uses_absolute_xdg_state_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("OAKAGENT_STORAGE_APP", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert (
        resolve_support_plane_path("inspection", "line-4")
        == tmp_path / "state" / "inspection" / "line-4" / "support-plane.json"
    )

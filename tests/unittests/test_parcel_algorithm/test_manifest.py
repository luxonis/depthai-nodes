"""The installed parcel bundle must agree with its frozen evidence."""

import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from depthai_nodes.algorithms import parcel
from depthai_nodes.algorithms.parcel.manifest import _validate

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "depthai_nodes/algorithms/parcel/manifest.json"


@pytest.fixture
def installed_copy(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    for name in manifest["implementation"]["files_sha256"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    return tmp_path, manifest


def test_manifest_validates_installed_bytes_and_evidence():
    manifest = parcel.validate_manifest()
    assert manifest["workbench_candidate"]["id"] == parcel.ALGORITHM_ID
    assert manifest["implementation"]["name"] == "fast_baseline"
    assert manifest["implementation"]["package_path"] == (
        "depthai_nodes/algorithms/parcel/implementation"
    )
    assert parcel.IMPLEMENTATION_NAME == "implementation"
    assert manifest["source"]["configuration_version"] == "v001"
    assert len(inspect.signature(parcel.create_components).parameters) == 0


def test_editing_computation_fails_integrity_check(installed_copy):
    root, manifest = installed_copy
    source = root / manifest["implementation"]["package_path"] / "geometry.py"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="File hash mismatch.*geometry.py"):
        _validate(root, manifest, check_bindings=False)


def test_shared_plane_normalization_is_part_of_the_covered_port(installed_copy):
    root, manifest = installed_copy
    name = "depthai_nodes/message/support_plane.py"
    assert name in manifest["implementation"]["files_sha256"]
    path = root / name
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="File hash mismatch.*support_plane.py"):
        _validate(root, manifest, check_bindings=False)


def test_omitted_file_and_added_computation_fail_coverage(installed_copy):
    root, manifest = installed_copy
    changed = copy.deepcopy(manifest)
    changed["implementation"]["files_sha256"].pop(
        next(iter(changed["implementation"]["files_sha256"]))
    )
    with pytest.raises(ValueError, match="Incomplete port file coverage"):
        _validate(root, changed, check_bindings=False)
    extra = root / manifest["implementation"]["package_path"] / "unrecorded.py"
    extra.write_text("pass\n")
    with pytest.raises(ValueError, match="Incomplete port file coverage"):
        _validate(root, manifest, check_bindings=False)


def test_frozen_configuration_cannot_change_with_port_hashes(installed_copy):
    root, manifest = installed_copy
    key = manifest["implementation"]["configuration_path"]
    path = root / key
    configuration = json.loads(path.read_text())
    configuration["detection"]["foreground_min_height_mm"] += 1
    path.write_text(json.dumps(configuration))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["implementation"]["files_sha256"][key] = digest
    manifest["implementation"]["configuration_sha256"] = digest
    with pytest.raises(ValueError, match="Frozen configuration changed"):
        _validate(root, manifest, check_bindings=False)


def test_candidate_and_suppressed_metrics_are_checked(installed_copy):
    root, manifest = installed_copy
    changed = copy.deepcopy(manifest)
    changed["workbench_candidate"]["id"] = "algorithm-other"
    with pytest.raises(ValueError, match="Candidate binding mismatch"):
        _validate(root, changed, check_bindings=False)
    test_entry = manifest["evaluations"][1]
    assert len(test_entry["suppressed_metrics"]) == 6
    for key in test_entry["suppressed_metrics"]:
        assert test_entry["metrics"][key]["status"] == "suppressed"
        assert test_entry["metrics"][key]["value"] is None
        assert test_entry["metrics"][key]["reason"]
    test_entry["metrics"][test_entry["suppressed_metrics"][0]]["value"] = 0
    with pytest.raises(ValueError, match="Evaluation test metrics mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_node_binding_mismatch_is_rejected(monkeypatch):
    from depthai_nodes.node.parcel_dimensioner import ParcelDimensioner

    monkeypatch.setattr(ParcelDimensioner, "algorithm_id", "algorithm-other")
    with pytest.raises(ValueError, match="Node algorithm binding mismatch"):
        parcel.validate_manifest()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("algorithm", "other", "Algorithm name mismatch"),
        ("configuration_version", "v002", "Configuration version mismatch"),
        ("settings_path", "settings.json", "Canonical settings path mismatch"),
        ("settings_file_sha256", "0" * 64, "Canonical settings byte hash mismatch"),
    ],
)
def test_source_configuration_identity_is_checked(
    installed_copy, field, value, message
):
    root, manifest = installed_copy
    manifest["source"][field] = value
    with pytest.raises(ValueError, match=message):
        _validate(root, manifest, check_bindings=False)


def test_visualization_is_shipped_and_covered(installed_copy):
    root, manifest = installed_copy
    name = manifest["visualization"]["path"]
    assert name in manifest["implementation"]["files_sha256"]
    page = root / name
    page.write_text(page.read_text() + "\n")
    with pytest.raises(ValueError, match="File hash mismatch.*visualization.html"):
        _validate(root, manifest, check_bindings=False)


def test_visualization_cannot_reassign_metrics_with_updated_port_hash(installed_copy):
    root, manifest = installed_copy
    name = manifest["visualization"]["path"]
    page = root / name
    text = page.read_text()
    text = text.replace(
        '"measurement_success_count":6', '"measurement_success_count":7'
    )
    page.write_text(text)
    digest = hashlib.sha256(page.read_bytes()).hexdigest()
    manifest["implementation"]["files_sha256"][name] = digest
    manifest["visualization"]["sha256"] = digest
    with pytest.raises(ValueError, match="Visualization metrics mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_public_node_export_binding_is_checked(monkeypatch):
    import depthai_nodes.node

    monkeypatch.setattr(depthai_nodes.node, "ParcelDimensioner", object)
    with pytest.raises(ValueError, match="Public node export binding mismatch"):
        parcel.validate_manifest()


def test_optional_scipy_does_not_prevent_type_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["scipy"] = None
from depthai_nodes.algorithms.parcel import PointCloud, create_components
try:
    create_components()
except ImportError as exc:
    assert 'depthai-nodes[parcel]' in str(exc)
else:
    raise AssertionError("Missing scipy must produce an installation hint")
""",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

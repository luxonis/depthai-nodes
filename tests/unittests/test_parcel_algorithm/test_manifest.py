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
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name in manifest["implementation"]["files_sha256"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    return tmp_path, manifest


def test_manifest_validates_installed_bytes_and_evidence():
    manifest = parcel.validate_manifest()
    assert manifest["workbench_candidate"]["id"] == parcel.ALGORITHM_ID
    assert manifest["implementation"]["name"] == "fast_baseline_oak4_visibility_01"
    assert manifest["implementation"]["package_path"] == (
        "depthai_nodes/algorithms/parcel/implementation"
    )
    assert parcel.IMPLEMENTATION_NAME == "implementation"
    assert manifest["source"]["configuration_version"] == "v002"
    assert manifest["implementation"]["configuration_path"] == (
        "depthai_nodes/algorithms/parcel/implementation/scoped-settings.json"
    )
    assert manifest["implementation"]["configuration_pointer"] == "/configuration"
    assert len(inspect.signature(parcel.create_components).parameters) == 0


def test_factory_receives_only_nested_frozen_configuration(monkeypatch):
    from depthai_nodes.algorithms.parcel import implementation

    settings_path = (
        ROOT / "depthai_nodes/algorithms/parcel/implementation/scoped-settings.json"
    )
    original = settings_path.read_bytes()
    expected = json.loads(original)["configuration"]
    received = []
    result = object()

    def create(configuration):
        received.append(configuration)
        return result

    monkeypatch.setattr(implementation, "create_components", create)
    assert parcel.create_components() is result
    assert received == [expected]
    assert settings_path.read_bytes() == original
    assert not settings_path.with_name("configuration.json").exists()


def test_configuration_pointer_cannot_change(installed_copy):
    root, manifest = installed_copy
    manifest["implementation"]["configuration_pointer"] = "/other"
    with pytest.raises(ValueError, match="Configuration pointer mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_manifest_validates_with_cp1252_default_encoding(monkeypatch):
    original_open = Path.open

    def cp1252_open(
        path, mode="r", buffering=-1, encoding=None, errors=None, newline=None
    ):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return original_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", cp1252_open)
    manifest = parcel.validate_manifest()
    assert manifest["workbench_candidate"]["id"] == parcel.ALGORITHM_ID


def test_editing_computation_fails_integrity_check(installed_copy):
    root, manifest = installed_copy
    source = root / manifest["implementation"]["package_path"] / "geometry.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="File hash mismatch.*geometry.py"):
        _validate(root, manifest, check_bindings=False)


def test_shared_plane_normalization_is_part_of_the_covered_port(installed_copy):
    root, manifest = installed_copy
    name = "depthai_nodes/message/support_plane.py"
    assert name in manifest["implementation"]["files_sha256"]
    path = root / name
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
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
    extra.write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete port file coverage"):
        _validate(root, manifest, check_bindings=False)


def test_frozen_configuration_cannot_change_with_port_hashes(installed_copy):
    root, manifest = installed_copy
    key = manifest["implementation"]["configuration_path"]
    path = root / key
    settings = json.loads(path.read_text(encoding="utf-8"))
    settings["configuration"]["detection"]["foreground_min_height_mm"] += 1
    path.write_text(json.dumps(settings), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["implementation"]["files_sha256"][key] = digest
    manifest["implementation"]["configuration_sha256"] = digest
    with pytest.raises(ValueError, match="Frozen settings byte hash mismatch"):
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


def test_native_source_archive_cannot_be_mislabeled_as_an_image(installed_copy):
    root, manifest = installed_copy
    candidate = manifest["workbench_candidate"]
    assert candidate["image_digest"] is None
    assert candidate["build_kind"] == "native-source-archive"
    candidate["image_digest"] = "sha256:" + candidate["build_sha256"]
    with pytest.raises(ValueError, match="Executed native build mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_review_must_cover_the_frozen_source(installed_copy):
    root, manifest = installed_copy
    manifest["source"]["review"]["source_commit"] = "0" * 40
    with pytest.raises(ValueError, match="Frozen source review mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_held_out_approval_matches_the_packaged_selected_candidate_receipt():
    manifest = parcel.validate_manifest()
    evidence = json.loads(
        (
            ROOT
            / manifest["implementation"]["package_path"]
            / "evaluation-evidence.json"
        ).read_bytes()
    )["held_out_result"]
    assert manifest["test_request"] == evidence["test_request"]
    assert manifest["test_request"]["id"] == evidence["run"]["test_request_id"]
    assert evidence["run"]["candidate"]["candidate_id"] == parcel.ALGORITHM_ID


def test_predecessor_approval_is_rejected(installed_copy):
    root, manifest = installed_copy
    manifest["test_request"].update(
        id="test-request-8bbaac21223541948afcfd16bb0c6c35",
        run_ids=["run-efd2b5483a4c452787a56f77e264e98e"],
        authorization_id="test-grant-c4a80185c19f4f54b72364796855a30a",
    )
    with pytest.raises(ValueError, match="Held-out approval receipt mismatch"):
        _validate(root, manifest, check_bindings=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "test-request-other"),
        ("status", "pending"),
        ("run_ids", ["run-other"]),
        ("authorization_id", "test-grant-other"),
        ("release_id", "release-other"),
        ("input_mode", "estimated_plane"),
        ("feedback_timing", "after_operator_release"),
    ],
)
def test_matching_receipt_still_requires_approval_run_linkage(
    installed_copy, field, value
):
    root, manifest = installed_copy
    name = manifest["implementation"]["package_path"] + "/evaluation-evidence.json"
    path = root / name
    evidence = json.loads(path.read_bytes())
    # Updating both declarations and their file hash cannot detach approval from
    # the selected candidate's saved run and frozen settings.
    evidence["held_out_result"]["test_request"][field] = value
    manifest["test_request"][field] = value
    path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest["implementation"]["files_sha256"][name] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="Held-out approval is not linked"):
        _validate(root, manifest, check_bindings=False)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("run", "candidate", {"candidate_id": "algorithm-predecessor"}),
        ("run", "evaluation_id", "evaluation-other"),
        ("evaluation", "run_id", "run-other"),
    ],
)
def test_held_out_candidate_and_evaluation_are_linked(
    installed_copy, section, field, value
):
    root, manifest = installed_copy
    name = manifest["implementation"]["package_path"] + "/evaluation-evidence.json"
    path = root / name
    evidence = json.loads(path.read_bytes())
    evidence["held_out_result"][section][field] = value
    path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest["implementation"]["files_sha256"][name] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(
        ValueError,
        match="Held-out approval is not linked|Evaluation candidate or settings mismatch",
    ):
        _validate(root, manifest, check_bindings=False)


@pytest.mark.parametrize(
    "field", ("binary_distribution", "native_build_flags", "tested_native_binaries")
)
def test_predecessor_native_build_claims_are_rejected(installed_copy, field):
    root, manifest = installed_copy
    historical = json.loads(
        (
            ROOT
            / "docs/parcel-promotions/historical-native-builds-algorithm-2f65a6f6296d48e981f5fe3f77bfde89.json"
        ).read_bytes()
    )
    assert len(historical["implementation"]["tested_native_binaries"]) == 6
    manifest["implementation"][field] = historical["implementation"][field]
    with pytest.raises(ValueError, match="Native build metadata does not describe"):
        _validate(root, manifest, check_bindings=False)


def test_numpy_candidate_has_no_obsolete_kernel_assets_or_imports():
    root = ROOT / "depthai_nodes/algorithms/parcel/implementation"
    assert not any(
        (root / name).exists()
        for name in ("native.py", "kernels.cpp", "build_native.py")
    )
    assert "numpy_ops.py" in {path.name for path in root.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from depthai_nodes.algorithms import parcel; import sys; parcel.create_components(); assert not any(name.endswith('.native') and ('parcel' in name or 'fast_baseline' in name) for name in sys.modules)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("algorithm", "other", "Algorithm name mismatch"),
        ("configuration_version", "v999", "Configuration version mismatch"),
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
    page.write_text(page.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="File hash mismatch.*visualization.html"):
        _validate(root, manifest, check_bindings=False)


def test_visualization_cannot_reassign_metrics_with_updated_port_hash(installed_copy):
    root, manifest = installed_copy
    name = manifest["visualization"]["path"]
    page = root / name
    text = page.read_text(encoding="utf-8")
    text = text.replace(
        '"measurement_success_count":8', '"measurement_success_count":9'
    )
    page.write_text(text, encoding="utf-8")
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

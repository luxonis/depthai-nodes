"""The installed MeasureAnything bundle must agree with its frozen evidence."""

import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from depthai_nodes.algorithms import measure_anything
from depthai_nodes.algorithms.measure_anything.manifest import _validate

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "depthai_nodes/algorithms/measure_anything/manifest.json"


@pytest.fixture
def installed_copy(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name in manifest["implementation"]["files_sha256"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    return tmp_path, manifest


def _evidence(root, manifest):
    return json.loads((root / manifest["source"]["evidence_path"]).read_bytes())


def _write_evidence(root, manifest, evidence):
    name = manifest["source"]["evidence_path"]
    path = root / name
    path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest["implementation"]["files_sha256"][name] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_manifest_validates_installed_bytes_and_evidence():
    manifest = measure_anything.validate_manifest()
    assert manifest["candidate"]["id"] == measure_anything.ALGORITHM_ID
    assert manifest["implementation"]["name"] == "fast_baseline_oak4_visibility_01"
    assert manifest["implementation"]["package_path"] == (
        "depthai_nodes/algorithms/measure_anything/implementation"
    )
    assert measure_anything.IMPLEMENTATION_NAME == "implementation"
    assert manifest["source"]["configuration_version"] == "v002"
    assert manifest["implementation"]["configuration_path"] == (
        "depthai_nodes/algorithms/measure_anything/implementation/scoped-settings.json"
    )
    assert manifest["implementation"]["configuration_pointer"] == "/configuration"
    assert len(inspect.signature(measure_anything.create_components).parameters) == 0


def test_factory_receives_only_nested_frozen_configuration(monkeypatch):
    from depthai_nodes.algorithms.measure_anything import implementation

    settings_path = (
        ROOT
        / "depthai_nodes/algorithms/measure_anything/implementation/scoped-settings.json"
    )
    original = settings_path.read_bytes()
    expected = json.loads(original)["configuration"]
    received = []
    result = object()

    def create(configuration):
        received.append(configuration)
        return result

    monkeypatch.setattr(implementation, "create_components", create)
    assert measure_anything.create_components() is result
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
    manifest = measure_anything.validate_manifest()
    assert manifest["candidate"]["id"] == measure_anything.ALGORITHM_ID


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
    changed["candidate"]["id"] = "algorithm-other"
    with pytest.raises(ValueError, match="Candidate binding mismatch"):
        _validate(root, changed, check_bindings=False)
    evidence = _evidence(root, manifest)
    evaluation = evidence["held_out_result"]["evaluation"]
    suppressed = evaluation["suppressed_metrics"]
    assert len(suppressed) == 6
    for key in suppressed:
        value = evaluation["metrics"]
        for part in key.split("."):
            value = value[part]
        assert value is None
        assert evaluation["metrics"]["null_reasons"][key]
    evaluation["metrics"]["dimension_errors_mm"]["height"]["median_absolute"] = 0
    _write_evidence(root, manifest, evidence)
    with pytest.raises(ValueError, match="Visualization metrics mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_node_binding_mismatch_is_rejected(monkeypatch):
    from depthai_nodes.node.measure_anything_dimensioner import (
        MeasureAnythingDimensioner,
    )

    monkeypatch.setattr(MeasureAnythingDimensioner, "algorithm_id", "algorithm-other")
    with pytest.raises(ValueError, match="Node algorithm binding mismatch"):
        measure_anything.validate_manifest()


def test_native_source_archive_cannot_be_mislabeled_as_an_image(installed_copy):
    root, manifest = installed_copy
    candidate = manifest["candidate"]
    assert candidate["image_digest"] is None
    assert candidate["build_kind"] == "native-source-archive"
    candidate["image_digest"] = "sha256:" + candidate["build_sha256"]
    with pytest.raises(ValueError, match="Executed native build mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_review_must_cover_the_frozen_source(installed_copy):
    root, manifest = installed_copy
    evidence = _evidence(root, manifest)
    evidence["review"]["source_commit"] = "0" * 40
    _write_evidence(root, manifest, evidence)
    with pytest.raises(ValueError, match="Frozen source review mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_held_out_approval_matches_the_packaged_selected_candidate_receipt():
    manifest = measure_anything.validate_manifest()
    held_out = _evidence(ROOT, manifest)["held_out_result"]
    request = held_out["test_request"]
    assert request["id"] == held_out["run"]["test_request_id"]
    assert request["feedback_released"] is False
    assert held_out["run"]["candidate"]["candidate_id"] == measure_anything.ALGORITHM_ID


def test_predecessor_approval_is_rejected(installed_copy):
    root, manifest = installed_copy
    evidence = _evidence(root, manifest)
    evidence["held_out_result"]["test_request"].update(
        id="test-request-8bbaac21223541948afcfd16bb0c6c35",
        run_ids=["run-efd2b5483a4c452787a56f77e264e98e"],
        authorization_id="test-grant-c4a80185c19f4f54b72364796855a30a",
    )
    _write_evidence(root, manifest, evidence)
    with pytest.raises(ValueError, match="Held-out approval is not linked"):
        _validate(root, manifest, check_bindings=False)


def test_evidence_cannot_be_replaced_by_updating_port_hashes(installed_copy):
    root, manifest = installed_copy
    evidence = _evidence(root, manifest)
    evidence["lineage_note"] += " changed"
    _write_evidence(root, manifest, evidence)
    with pytest.raises(
        ValueError, match="Frozen evaluation evidence byte hash mismatch"
    ):
        _validate(root, manifest, check_bindings=False)


def test_evidence_path_cannot_point_outside_the_selected_package(installed_copy):
    root, manifest = installed_copy
    manifest["source"]["evidence_path"] = "../other-evidence.json"
    with pytest.raises(ValueError, match="Evidence path mismatch"):
        _validate(root, manifest, check_bindings=False)


@pytest.mark.parametrize("field", ("path", "url"))
def test_comparison_reference_is_pinned(installed_copy, field):
    root, manifest = installed_copy
    manifest["port_verification"]["comparison"][field] = "other.json"
    with pytest.raises(ValueError, match="Comparison reference mismatch"):
        _validate(root, manifest, check_bindings=False)


@pytest.mark.parametrize("change", ("unrecorded", "missing", "swapped"))
def test_source_to_port_mapping_is_complete_and_correct(installed_copy, change):
    root, manifest = installed_copy
    mapping = manifest["source"]["source_to_port"]
    if change == "unrecorded":
        mapping["unrecorded.py"] = "geometry.py"
    elif change == "missing":
        mapping.clear()
    else:
        mapping["geometry.py"], mapping["visibility.py"] = (
            mapping["visibility.py"],
            mapping["geometry.py"],
        )
    with pytest.raises(ValueError, match="Source-to-port mapping mismatch"):
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
def test_approval_receipt_requires_matching_run(installed_copy, field, value):
    root, manifest = installed_copy
    evidence = _evidence(root, manifest)
    evidence["held_out_result"]["test_request"][field] = value
    _write_evidence(root, manifest, evidence)
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
    evidence = _evidence(root, manifest)
    evidence["held_out_result"][section][field] = value
    _write_evidence(root, manifest, evidence)
    with pytest.raises(
        ValueError,
        match="Held-out approval is not linked|Evaluation candidate or settings mismatch",
    ):
        _validate(root, manifest, check_bindings=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("binary_distribution", "Custom C++ kernels built locally"),
        ("native_build_flags", ["-std=c++17", "-O3"]),
        ("tested_native_binaries", [{"path": "native.so", "sha256": "0" * 64}]),
    ],
)
def test_predecessor_native_build_claims_are_rejected(installed_copy, field, value):
    root, manifest = installed_copy
    manifest["implementation"][field] = value
    with pytest.raises(ValueError, match="Native build metadata does not describe"):
        _validate(root, manifest, check_bindings=False)


def test_numpy_candidate_has_no_obsolete_kernel_assets_or_imports():
    root = ROOT / "depthai_nodes/algorithms/measure_anything/implementation"
    assert not any(
        (root / name).exists()
        for name in ("native.py", "kernels.cpp", "build_native.py")
    )
    assert "numpy_ops.py" in {path.name for path in root.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from depthai_nodes.algorithms import measure_anything; import sys; measure_anything.create_components(); assert not any(name.endswith('.native') and ('measure_anything' in name or 'fast_baseline' in name) for name in sys.modules)",
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


def test_public_visualization_identifies_the_selected_reference_artifact(
    installed_copy,
):
    root, manifest = installed_copy
    manifest["visualization"]["reference_sha256"] = "0" * 64
    with pytest.raises(
        ValueError, match="Public visualization export provenance mismatch"
    ):
        _validate(root, manifest, check_bindings=False)


def test_public_evidence_identifies_the_selected_reference_artifact(installed_copy):
    root, manifest = installed_copy
    evidence = _evidence(root, manifest)
    evidence["public_export"]["reference_evidence_sha256"] = "0" * 64
    _write_evidence(root, manifest, evidence)
    with pytest.raises(ValueError, match="Public evidence export provenance mismatch"):
        _validate(root, manifest, check_bindings=False)


def test_visualization_uses_existing_bundled_links(installed_copy):
    import re

    root, manifest = installed_copy
    manifest_path = root / "depthai_nodes/algorithms/measure_anything/manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    page = root / manifest["visualization"]["path"]
    static_html = page.read_text(encoding="utf-8").split("<script", 1)[0]
    for href in re.findall(r'href="([^"]+)"', static_html):
        if href.startswith("#"):
            continue
        target = (page.parent / href).resolve()
        assert target.is_relative_to((root / "depthai_nodes").resolve())
        assert target.is_file(), href


def test_exported_policy_digest_covers_public_scoring_parameters(installed_copy):
    root, manifest = installed_copy
    evidence = _evidence(root, manifest)
    evidence["validation"]["run"]["policy_sha256"] = "0" * 64
    _write_evidence(root, manifest, evidence)
    with pytest.raises(ValueError, match="Public scoring policy digest mismatch"):
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

    monkeypatch.setattr(depthai_nodes.node, "MeasureAnythingDimensioner", object)
    with pytest.raises(ValueError, match="Public node export binding mismatch"):
        measure_anything.validate_manifest()


def test_optional_scipy_does_not_prevent_type_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["scipy"] = None
from depthai_nodes.algorithms.measure_anything import PointCloud, create_components
try:
    create_components()
except ImportError as exc:
    assert 'depthai-nodes[measure-anything]' in str(exc)
else:
    raise AssertionError("Missing scipy must produce an installation hint")
""",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

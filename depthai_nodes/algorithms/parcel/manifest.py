"""Integrity checks for the installed parcel implementation.

File keys are repository-relative paths beginning with ``depthai_nodes/``. Hashes
cover raw installed bytes and exclude this self-describing manifest. The selected implementation uses no custom native kernels or binaries.
"""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_COMMIT = "d5e35e17d75f91368889dc53d950792eed4096e6"
_EVIDENCE_COMMIT = "d5964eee714737750787066fc24bc29744536d45"
_BINARY_DISTRIBUTION = (
    "No custom native kernels, sources, build flags, or binaries; "
    "computation uses NumPy, SciPy, and OpenCV."
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: str) -> str:
    _require(
        isinstance(value, str) and bool(_DIGEST.fullmatch(value)), "Invalid SHA-256"
    )
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _metric_records(evaluation: dict) -> dict:
    metrics = evaluation["metrics"]
    suppressed = evaluation["suppressed_metrics"]
    records = {}
    for name, value in metrics.items():
        if name in {"denominators", "null_reasons"}:
            continue
        if name == "dimension_errors_mm":
            for axis, errors in value.items():
                for statistic, number in errors.items():
                    key = f"dimension_errors_mm.{axis}.{statistic}"
                    records[key] = {
                        "status": "suppressed" if key in suppressed else "available",
                        "value": number,
                        "unit": "count"
                        if statistic == "count"
                        else "ratio"
                        if statistic == "mean_relative_absolute"
                        else "mm",
                        "denominator": errors["count"],
                        "reason": metrics["null_reasons"].get(key),
                    }
        else:
            records[name] = {
                "status": "available" if value is not None else "not_reported",
                "value": value,
                "unit": "count"
                if name.endswith("_count") or name in {"tp", "fp", "fn"}
                else "ratio",
                "denominator": metrics["denominators"].get(name),
                "reason": metrics["null_reasons"].get(name)
                or (
                    "Not reported by the retained supplied-plane evaluation"
                    if value is None
                    else None
                ),
            }
    return records


def _covered_files(root: Path) -> set[str]:
    package = root / "depthai_nodes"
    paths = set()
    directories = [
        package / "algorithms" / "parcel",
        package / "node" / "parcel_detector",
        package / "node" / "parcel_dimensioner",
    ]
    for directory in directories:
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".so", ".dylib", ".dll", ".pyd"}
                and path.name != "manifest.json"
            ):
                paths.add(path.relative_to(root).as_posix())
    paths.add("depthai_nodes/algorithms/__init__.py")
    paths.add("depthai_nodes/message/support_plane.py")
    for path in (package / "message").glob("parcel*.py"):
        paths.add(path.relative_to(root).as_posix())
    return paths


def _validate(root: Path, manifest: dict, *, check_bindings: bool = True) -> dict:
    from . import (
        _ALGORITHM_NAME,
        _CONFIGURATION_VERSION,
        ALGORITHM_ID,
        IMPLEMENTATION_NAME,
    )

    _require(
        manifest["schema_version"] == "parcel-algorithm-manifest/1",
        "Unknown manifest schema",
    )
    implementation = manifest["implementation"]
    package_path = f"depthai_nodes/algorithms/parcel/{IMPLEMENTATION_NAME}"
    _require(
        implementation["name"] == _ALGORITHM_NAME, "Implementation binding mismatch"
    )
    _require(
        implementation["package_path"] == package_path, "Implementation path mismatch"
    )
    _require(
        implementation["configuration_path"] == package_path + "/configuration.json",
        "Configuration path mismatch",
    )
    _require(
        implementation["source_contract_version"]
        == f"BoxLabRunners@{_SOURCE_COMMIT}:docs/boxlab/contracts/RUNNER.md",
        "Source contract mismatch",
    )
    _require(
        implementation["binary_distribution"] == _BINARY_DISTRIBUTION
        and implementation["native_build_flags"] == []
        and implementation["tested_native_binaries"] == [],
        "Native build metadata does not describe the selected NumPy implementation",
    )
    files = implementation["files_sha256"]
    _require(
        set(files) == _covered_files(root),
        "Incomplete port file coverage",
    )
    for name, expected in files.items():
        path = PurePosixPath(name)
        _require(
            path.parts[0] == "depthai_nodes"
            and not path.is_absolute()
            and ".." not in path.parts,
            "Unsafe manifest file path",
        )
        actual_path = root.joinpath(*path.parts)
        _require(
            actual_path.resolve().is_relative_to((root / "depthai_nodes").resolve()),
            "Manifest path escapes package",
        )
        _require(
            hashlib.sha256(actual_path.read_bytes()).hexdigest() == _digest(expected),
            f"File hash mismatch: {name}",
        )
    configuration_bytes = (root / implementation["configuration_path"]).read_bytes()
    _require(
        hashlib.sha256(configuration_bytes).hexdigest()
        == _digest(implementation["configuration_sha256"]),
        "Configuration byte hash mismatch",
    )
    configuration = json.loads(configuration_bytes)
    settings_bytes = (root / package_path / "scoped-settings.json").read_bytes()
    settings = json.loads(settings_bytes)
    evidence = json.loads(
        (root / package_path / "evaluation-evidence.json").read_bytes()
    )
    _require(
        hashlib.sha256(settings_bytes).hexdigest() == evidence["settings_sha256"],
        "Frozen settings byte hash mismatch",
    )
    _require(configuration == settings["configuration"], "Frozen configuration changed")
    candidate = manifest["workbench_candidate"]
    original = evidence["validation"]["run"]["candidate"]
    _require(
        candidate["id"] == original["candidate_id"] == ALGORITHM_ID,
        "Candidate binding mismatch",
    )
    for field in ("code_sha256", "configuration_sha256", "dependencies_sha256"):
        _require(
            _digest(candidate[field]) == original[field],
            f"Original candidate {field} mismatch",
        )
    _require(
        hashlib.sha256(
            _canonical(
                {"algorithm": settings["algorithm"], "parameters": configuration}
            )
        ).hexdigest()
        == candidate["configuration_sha256"],
        "Canonical configuration hash mismatch",
    )
    _require(
        candidate["weight_sha256s"] == original["weight_sha256s"] == [],
        "Unexpected weights",
    )
    _require(
        candidate["image_digest"] is None
        and evidence["image"] is None
        and candidate["build_kind"]
        == evidence["build"]["kind"]
        == "native-source-archive"
        and candidate["build_sha256"]
        == original["build_sha256"]
        == evidence["build"]["artifact"]["sha256"],
        "Executed native build mismatch",
    )
    source = manifest["source"]
    _require(source["algorithm"] == _ALGORITHM_NAME, "Algorithm name mismatch")
    _require(
        source["configuration_version"] == _CONFIGURATION_VERSION,
        "Configuration version mismatch",
    )
    _require(
        settings["algorithm"] == f"algorithms.{_ALGORITHM_NAME}",
        "Settings algorithm mismatch",
    )
    _require(
        source["settings_path"]
        == f"algorithms/{_ALGORITHM_NAME}/configurations/{_CONFIGURATION_VERSION}.json",
        "Canonical settings path mismatch",
    )
    _require(
        source["settings_file_sha256"] == evidence["settings_sha256"],
        "Canonical settings byte hash mismatch",
    )
    _require(
        source["commit"] == evidence["source_commit"] == _SOURCE_COMMIT,
        "Frozen source revision mismatch",
    )
    _require(
        source["evidence_commit"] == _EVIDENCE_COMMIT, "Evidence revision mismatch"
    )
    _require(
        source["original_files_sha256"]
        == evidence["provenance"]["source_files_sha256"],
        "Original source hashes mismatch",
    )
    _require(
        set(source["paths"]) == set(source["original_files_sha256"]),
        "Original source coverage mismatch",
    )
    for digest in source["original_files_sha256"].values():
        _digest(digest)
    _require(source["lineage_note"] == evidence["lineage_note"], "Lineage mismatch")
    review = source["review"]
    _require(
        review == evidence["review"]
        and review["source_commit"] == source["commit"]
        and bool(review["provider"].strip())
        and review["url"] == "https://github.com/luxonis/BoxLabRunners/pull/19"
        and all(finding.get("resolution") for finding in review["findings"]),
        "Frozen source review mismatch",
    )
    visualization = manifest["visualization"]
    _require(
        visualization["path"] == package_path + "/visualization.html",
        "Visualization path mismatch",
    )
    _require(
        visualization["sha256"] == files[visualization["path"]],
        "Visualization file hash mismatch",
    )
    _require(
        visualization["source_repository"] == source["repository"]
        and bool(re.fullmatch(r"[0-9a-f]{40}", visualization["source_commit"]))
        and visualization["source_path"]
        == f"algorithms/{_ALGORITHM_NAME}/visualization.html",
        "Visualization source mismatch",
    )
    _digest(visualization["source_sha256"])
    html = (root / visualization["path"]).read_text(encoding="utf-8")
    match = re.search(
        r'<script id="evidence-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    _require(match is not None, "Missing visualization evidence")
    displayed = json.loads(match[1])
    _require(
        displayed["candidate"] == original
        and displayed["configuration"] == configuration
        and displayed["source_commit"] == source["commit"]
        and displayed["candidate"]["build_sha256"] == candidate["build_sha256"],
        "Visualization candidate or configuration mismatch",
    )
    for key, target in (("validation", "validation"), ("test", "held_out_result")):
        _require(
            (
                displayed[key]["evaluation"]["metrics"]
                if key == "validation"
                else displayed[key]["metrics"]
            )
            == evidence[target]["evaluation"]["metrics"],
            "Visualization metrics mismatch",
        )
    inputs = manifest["inputs"]
    _require(
        set(inputs["modalities"]) == set(settings["execution"]["modalities"]),
        "Input modalities mismatch",
    )
    _require(
        inputs["input_mode"] == settings["input_mode"] == "supplied_plane",
        "Plane mode mismatch",
    )
    _require(
        inputs["length_unit"] == "mm" and inputs["plane_orientation"] == "upward",
        "Scene conventions mismatch",
    )
    _require(
        inputs["frozen_runtime"] == evidence["provenance"]["runtime"],
        "Frozen runtime mismatch",
    )
    request = manifest["test_request"]
    held_out = evidence["held_out_result"]
    test_run = held_out["run"]
    _require(
        request == held_out["test_request"],
        "Held-out approval receipt mismatch",
    )
    _require(
        request["schema_version"] == "boxlab-test-request/1"
        and request["status"] == "approved"
        and bool(request["id"])
        and request["id"] == test_run["test_request_id"]
        and isinstance(request["run_ids"], list)
        and test_run["id"] in request["run_ids"]
        and bool(request["authorization_id"])
        and request["authorization_id"] == test_run["authorization_id"]
        and request["release_id"]
        == test_run["release_id"]
        == evidence["validation"]["run"]["release_id"]
        and request["input_mode"] == test_run["input_mode"] == settings["input_mode"]
        and request["feedback_timing"] == settings["feedback_timing"]
        and test_run["candidate"] == original,
        "Held-out approval is not linked to the selected candidate and run",
    )
    _require(len(manifest["evaluations"]) == 2, "Expected validation and test evidence")
    for entry, key in zip(manifest["evaluations"], ("validation", "held_out_result")):
        run, evaluation = evidence[key]["run"], evidence[key]["evaluation"]
        _require(
            run["candidate"] == original
            and run["split"] == ("validation" if key == "validation" else "test")
            and evaluation["id"] == run["evaluation_id"]
            and evaluation["run_id"] == run["id"]
            and run["input_mode"] == settings["input_mode"]
            and run["execution"] == evidence["validation"]["run"]["execution"]
            and run["policy"] == evidence["validation"]["run"]["policy"]
            and evaluation["policy"] == run["policy"],
            "Evaluation candidate or settings mismatch",
        )
        expected = {
            "split": run["split"],
            "release_id": run["release_id"],
            "run_id": run["id"],
            "evaluation_id": evaluation["id"],
            "run_status": run["status"],
            "evaluation_status": evaluation["status"],
            "expected_count": run["expected_count"],
            "accepted_count": run["accepted_count"],
            "failed_snapshot_count": evaluation["metrics"]["failed_snapshot_count"],
            "metric_version": evaluation["metric_version"],
            "policy": evaluation["policy"],
            "policy_sha256": run["policy_sha256"],
            "objective": settings["objective"],
            "feedback_status": evaluation["feedback"],
            "suppressed_metrics": evaluation["suppressed_metrics"],
            "raw_metrics": evaluation["metrics"],
            "metrics": _metric_records(evaluation),
        }
        for field, value in expected.items():
            _require(
                entry[field] == value, f"Evaluation {run['split']} {field} mismatch"
            )
        _require(
            entry["run_status"] == "sealed"
            and entry["evaluation_status"] == "completed"
            and entry["accepted_count"] == entry["expected_count"],
            "Incomplete frozen evaluation",
        )
    verification = manifest["port_verification"]
    _require(
        verification["record_path"].startswith("docs/parcel-promotions/"),
        "Missing promotion record",
    )
    _require(
        verification["reference_artifact"] == original["build_sha256"],
        "Port reference mismatch",
    )
    for field in (
        "comparison_inputs",
        "numerical_tolerance",
        "checks",
        "environment",
        "limitations",
    ):
        _require(field in verification, f"Missing port verification {field}")
    if check_bindings:
        from depthai_nodes.node import ParcelDetector as PublicDetector
        from depthai_nodes.node import ParcelDimensioner as PublicDimensioner
        from depthai_nodes.node.parcel_detector import ParcelDetector
        from depthai_nodes.node.parcel_dimensioner import ParcelDimensioner

        _require(
            ParcelDetector.algorithm_id
            == ParcelDimensioner.algorithm_id
            == ALGORITHM_ID,
            "Node algorithm binding mismatch",
        )
        _require(
            PublicDetector is ParcelDetector and PublicDimensioner is ParcelDimensioner,
            "Public node export binding mismatch",
        )
    return manifest


def validate_manifest() -> dict:
    """Validate source identity, evaluation provenance, file hashes, and bindings.

    Returns:
        The manifest after successful validation.

    Raises:
        ValueError: If provenance, file coverage, or either node binding differs.
    """
    root = Path(__file__).resolve().parents[3]
    try:
        manifest = json.loads(
            Path(__file__).with_name("manifest.json").read_text(encoding="utf-8")
        )
        return _validate(root, manifest)
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid parcel manifest: {exc}") from exc

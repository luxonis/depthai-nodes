"""Integrity checks for the installed MeasureAnything implementation.

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
_REFERENCE_VISUALIZATION_SHA256 = (
    "c195176e3d77e4df433ea824bb219c7a91ef17412792635baf106ef6c6a2ae32"
)
_REFERENCE_EVIDENCE_SHA256 = (
    "e23bcd5d41638487409b52dcce3ace3bb024c954c4cae6a65307f235fbc0a9d8"
)
_EVIDENCE_SHA256 = "eca5fb520bac16dd504ff3c3a4c5b33ffba0e1b3cc6d84b14662cdb5e24f80bf"
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


def _covered_files(root: Path) -> set[str]:
    package = root / "depthai_nodes"
    paths = set()
    directories = [
        package / "algorithms" / "measure_anything",
        package / "node" / "measure_anything_detector",
        package / "node" / "measure_anything_dimensioner",
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
    for path in (package / "message").glob("measure_anything*.py"):
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
        manifest["schema_version"] == "measure-anything-algorithm-manifest/3",
        "Unknown manifest schema",
    )
    implementation = manifest["implementation"]
    package_path = f"depthai_nodes/algorithms/measure_anything/{IMPLEMENTATION_NAME}"
    _require(
        implementation["name"] == _ALGORITHM_NAME, "Implementation binding mismatch"
    )
    _require(
        implementation["package_path"] == package_path, "Implementation path mismatch"
    )
    _require(
        implementation["configuration_path"] == package_path + "/scoped-settings.json",
        "Configuration path mismatch",
    )
    _require(
        implementation["configuration_pointer"] == "/configuration",
        "Configuration pointer mismatch",
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
    settings_bytes = (root / implementation["configuration_path"]).read_bytes()
    _require(
        hashlib.sha256(settings_bytes).hexdigest()
        == _digest(implementation["configuration_sha256"]),
        "Configuration byte hash mismatch",
    )
    settings = json.loads(settings_bytes)
    configuration = settings["configuration"]
    source = manifest["source"]
    _require(
        source["evidence_path"] == package_path + "/evaluation-evidence.json",
        "Evidence path mismatch",
    )
    evidence_bytes = (root / source["evidence_path"]).read_bytes()
    evidence = json.loads(evidence_bytes)
    _require(
        hashlib.sha256(settings_bytes).hexdigest() == evidence["settings_sha256"],
        "Frozen settings byte hash mismatch",
    )
    candidate = manifest["candidate"]
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
    _require(source["publisher"] == "Luxonis", "Reference publisher mismatch")
    _require(
        evidence["public_export"]["schema_version"] == "public-evidence-export/1"
        and evidence["public_export"]["reference_evidence_sha256"]
        == _REFERENCE_EVIDENCE_SHA256,
        "Public evidence export provenance mismatch",
    )
    original_files = evidence["provenance"]["source_files_sha256"]
    for digest in original_files.values():
        _digest(digest)
    prefix = f"algorithms/{_ALGORITHM_NAME}/"
    expected_ports = {}
    for name in original_files:
        if name.startswith(prefix):
            original_name = name.removeprefix(prefix)
            expected_ports[original_name] = (
                "factory.py" if original_name == "__init__.py" else original_name
            )
    _require(
        source["source_to_port"] == expected_ports,
        "Source-to-port mapping mismatch",
    )
    for original_name, port_name in source["source_to_port"].items():
        _require(
            f"algorithms/{_ALGORITHM_NAME}/{original_name}" in original_files
            and package_path + "/" + port_name in files,
            "Source-to-port mapping mismatch",
        )
    review = evidence["review"]
    _require(
        review["source_commit"] == source["commit"]
        and bool(review["provider"].strip())
        and review["status"] == "completed"
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
        visualization["source_publisher"] == source["publisher"]
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
        displayed["public_export"]["schema_version"] == "public-evidence-export/1"
        and displayed["public_export"]["reference_evidence_sha256"]
        == _REFERENCE_EVIDENCE_SHA256
        and displayed["public_export"]["reference_visualization_sha256"]
        == visualization["reference_sha256"]
        == _REFERENCE_VISUALIZATION_SHA256,
        "Public visualization export provenance mismatch",
    )
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
    held_out = evidence["held_out_result"]
    request = held_out["test_request"]
    test_run = held_out["run"]
    _require(
        request["schema_version"] == "measurement-test-request/1"
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
    for key in ("validation", "held_out_result"):
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
        _require(
            hashlib.sha256(_canonical(run["policy"])).hexdigest()
            == run["policy_sha256"],
            "Public scoring policy digest mismatch",
        )
        _require(
            run["status"] == "sealed"
            and evaluation["status"] == "completed"
            and run["accepted_count"] == run["expected_count"],
            "Incomplete frozen evaluation",
        )
    _require(
        hashlib.sha256(evidence_bytes).hexdigest() == _EVIDENCE_SHA256,
        "Frozen evaluation evidence byte hash mismatch",
    )
    verification = manifest["port_verification"]
    _require(
        verification["record_path"].startswith("docs/measure-anything/"),
        "Missing promotion record",
    )
    _require(
        verification["reference_artifact"] == original["build_sha256"],
        "Port reference mismatch",
    )
    comparison = verification["comparison"]
    _require(
        comparison["path"] == "docs/measure-anything/comparison.json"
        and bool(
            re.fullmatch(
                r"https://github\.com/luxonis/depthai-nodes/blob/[0-9a-f]{40}/"
                + re.escape(comparison["path"]),
                comparison["url"],
            )
        ),
        "Comparison reference mismatch",
    )
    _digest(comparison["sha256"])
    _require(
        bool(verification["adaptations"])
        and all(
            isinstance(value, str) and value.strip()
            for value in verification["adaptations"]
        ),
        "Missing port adaptations",
    )
    if check_bindings:
        from depthai_nodes.node import MeasureAnythingDetector as PublicDetector
        from depthai_nodes.node import MeasureAnythingDimensioner as PublicDimensioner
        from depthai_nodes.node.measure_anything_detector import MeasureAnythingDetector
        from depthai_nodes.node.measure_anything_dimensioner import (
            MeasureAnythingDimensioner,
        )

        _require(
            MeasureAnythingDetector.algorithm_id
            == MeasureAnythingDimensioner.algorithm_id
            == ALGORITHM_ID,
            "Node algorithm binding mismatch",
        )
        _require(
            PublicDetector is MeasureAnythingDetector
            and PublicDimensioner is MeasureAnythingDimensioner,
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
        raise ValueError(f"Invalid MeasureAnything manifest: {exc}") from exc

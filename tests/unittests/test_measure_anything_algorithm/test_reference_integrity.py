import json

import pytest

from tests.measure_anything_reference import verify_measure_anything_port as verifier


@pytest.fixture
def reference(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    root = tmp_path / "reference"
    root.mkdir()
    name = "algorithms/fast_baseline/geometry.py"
    source = root / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"FROZEN = True\n")
    settings = root / "fast-baseline-settings.json"
    settings.write_text(json.dumps({"configuration": {"size_mm": 12}}))
    evidence = {
        "source_commit": verifier.SOURCE_COMMIT,
        "provenance": {
            "source_files_sha256": {name: verifier.digest(source.read_bytes())}
        },
        "settings_sha256": verifier.digest(settings.read_bytes()),
    }
    bundled = repository.joinpath(*verifier.PACKAGE.split("."))
    bundled.mkdir(parents=True)
    (bundled / "evaluation-evidence.json").write_text(json.dumps(evidence))
    (root / "fast-baseline.json").write_text(json.dumps(evidence))
    monkeypatch.setattr(verifier, "REPO_ROOT", repository)
    return root, source, settings, evidence


def test_matching_reference_is_accepted(reference):
    root, _, settings, evidence = reference
    assert verifier.verify_source(root) == (evidence, json.loads(settings.read_bytes()))


@pytest.mark.parametrize("changed", ("source", "settings"))
def test_changed_reference_cannot_redeclare_its_own_hashes(reference, changed):
    root, source, settings, evidence = reference
    if changed == "source":
        source.write_bytes(b"FROZEN = False\n")
        name = source.relative_to(root).as_posix()
        evidence["provenance"]["source_files_sha256"][name] = verifier.digest(
            source.read_bytes()
        )
    else:
        settings.write_text(json.dumps({"configuration": {"size_mm": 24}}))
        evidence["settings_sha256"] = verifier.digest(settings.read_bytes())
    (root / "fast-baseline.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="Reference evidence differs"):
        verifier.verify_source(root)


@pytest.mark.parametrize("changed", ("source", "settings"))
def test_changed_reference_bytes_are_rejected(reference, changed):
    root, source, settings, _ = reference
    path = source if changed == "source" else settings
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(
        ValueError, match="Frozen reference source|Frozen scoped settings"
    ):
        verifier.verify_source(root)

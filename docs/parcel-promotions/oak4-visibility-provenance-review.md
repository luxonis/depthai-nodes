# Visibility v002 provenance review fixes

Review of PR #348 at `47e0646e4ddef6de4f9d558e51f176ef4387858f` found
predecessor metadata left in the active manifest. These corrections change
metadata and integrity validation only; computation, frozen configuration,
source/candidate identity, original receipts, scores, and visualization are unchanged.

## Held-out approval

The manifest now copies the selected candidate's packaged `test_request` receipt
exactly, including its saved `feedback_released: false` observation. Available
evaluation feedback is not permission to rewrite that historical observation.

- Candidate: `algorithm-5e4abc3a09b54148950317cdeac60ba2`.
- Request: `test-request-058dab05ac1a49d7a838469670eb611d`.
- Run: `run-5cb93153a4cc436baad5ecaea3d8670a`.
- Authorization: `test-grant-6bc4742fc6e044a1a78149edb5418e1e`.
- Evaluation: `evaluation-6ce5ecda7afb46c8bb735b590bec0283`.

Validation checks the complete manifest receipt against the packaged receipt, then
links the approved request to the saved run, full candidate identity, authorization,
release, supplied-plane mode, and frozen feedback timing. Both evaluation records
must also link their evaluation IDs and run IDs to the appropriate split.
Regressions update both the receipt and manifest (and their byte hash) together
to verify that internally matching declarations cannot detach approval from the
selected candidate's run.

## Native-build metadata

The active NumPy/SciPy/OpenCV implementation declares no custom kernels, source
builds, flags, or binaries. Native flags and tested-binary lists are empty. The
validator rejects the predecessor's distribution text, flags, or binary list
when reintroduced independently.

The [historical build record](historical-native-builds-algorithm-2f65a6f6296d48e981f5fe3f77bfde89.json)
retains the original distribution statement, C++17 flags, and all six predecessor
binary records verbatim, under candidate
`algorithm-2f65a6f6296d48e981f5fe3f77bfde89`. Its source manifest is pinned to
`db6c4e780679b4be52f5adcc25d581c70d593928`. No predecessor timing or binary
verification is reassigned to the selected visibility candidate.

## Verification

The old validator accepts the predecessor approval and each of the three stale
native-build fields; the corrected validator rejects every reproduced case.
The focused parcel/reference/box run passes 85 tests and 46 subtests, with one
existing optional Open3D skip. Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  tests/unittests/test_parcel_algorithm tests/unittests/test_parcel_nodes \
  tests/parcel_reference \
  tests/unittests/test_nodes/test_threaded_host_nodes/test_box_detector_node.py \
  tests/unittests/test_nodes/test_threaded_host_nodes/test_box_measurer_node.py \
  tests/unittests/test_nodes/test_threaded_host_nodes/test_box_nodes_without_open3d.py -q
pre-commit run --all-files
pydoctor depthai_nodes
python -m build --wheel --no-isolation
```

The [review-fix wheel receipt](oak4-visibility-provenance-fix-wheel.json) is recorded separately from the original promotion wheel. A fresh installed wheel validates all 23 covered files, rejects the stale metadata, and passes 50 native DepthAI checks and a one-parcel synthetic pipeline. Only `manifest.json` and `manifest.py` differ from the original wheel package; computation, settings, evaluation receipts, and the visualization are byte-identical. Pre-commit and API documentation pass. No Workbench evaluation, held-out input access, or new hardware-performance
measurement is needed for these provenance-only corrections. Previous full-suite
failures and platform limitations remain recorded in the original promotion record.

# Shared axial-depth correction handoff

Verified on November 24, 2026. The implementation targets DepthAI 3.12 with the
exact initial dependency pin `depthai==3.12.0`.

## Package identity

| Field                             | Value                                                                      |
| --------------------------------- | -------------------------------------------------------------------------- |
| PR base                           | `codex/parcel-fast-baseline` at `51b6897d4efca7b6ada5400e05e009b3e57c495b` |
| Implementation revision           | `6ef43a9a2fcd40480976232121ca7077256aee86`                                 |
| Wheel                             | `depthai_nodes-0.8.0-py3-none-any.whl`                                     |
| Wheel SHA-256                     | `4b4dc99309d56a8379809c729d7f97b413ec6fa60fcd7b9a875eb9dde5ac9760`         |
| Installed package content SHA-256 | `f3531040550d973e2e3f588cd82ab18b0a9cbdd13c71e642ccf54a9083a05f40`         |
| SDK                               | `3.12.0`                                                                   |
| Example dependency pins           | NumPy `2.2.6`, OpenCV headless `4.12.0.88`                                 |
| Wheel build timestamp             | `SOURCE_DATE_EPOCH=1795528800`                                             |

The package version remains `0.8.0` according to the base branch. Pin the
revision and wheel digest above. A version string alone does not identify this
unreleased artifact. The delivered evidence bundle contains the wheel, host
and OAK-4 JSON records, logs, checksums, and the exact fixture source.

## Exact public API

The [API reference](axial-depth-correction.md) describes the complete contract.
The entrypoints are:

- `depthai_nodes.utils.AxialDepthPolicy` with
  `SOURCE_AXIAL_REQUIRES_CORRECTION` and `TARGET_AXIAL_PASSTHROUGH`.
- `depthai_nodes.utils.AxialDepthGeometry(target_size, target_intrinsics, source_from_target_mm, identity=())`.
- `depthai_nodes.utils.geometry_from_frames(aligned_frame, source_frame)`.
- `depthai_nodes.utils.AxialDepthCorrector(*, policy, cache_entries=4, cache_bytes=67108864)`.
- `corrector.correct(depth, geometry, *, previous_provenance=None)` returns
  `AxialDepthResult(depth, provenance)`.
- `corrector.correct_frame(aligned_frame, source_frame, *, previous_provenance=None)` returns `AxialDepthFrameResult(frame, provenance)`.
- `depthai_nodes.node.ImageAlignDepthCorrection.build(alignedDepth, sourceDepth, *, policy)` connects one ImageAlign's `outputAligned` and
  `passthroughInput`. `out` supplies native RAW16 depth. `result` supplies the
  frame and provenance in one same-runtime message.

These helper types are also available from `depthai_nodes.utils.axial_depth`.
`AxialDepthFrameResult` is exported from `depthai_nodes.message`.

The benchmark must choose `source_axial_requires_correction` for the verified
3.12.0 ImageAlign path and record `provenance.to_dict()` for each occurrence.
It reports `correction_enabled=true`, method
`source_axial_to_target_axial/1`, input `source_camera_axial_z`, and output
`target_camera_axial_z`. Use the actual rectified depth passthrough metadata.
The aligned transformation supplies the target image grid and pose.

`target_axial_passthrough` explicitly declares an already target-axial
producer. No SDK-version heuristic selects it. Structured results reject
double correction. Native RAW16 output needs its retained provenance at any
subsequent correction boundary because the SDK has no durable convention tag.

## Verification results

| Check                                                  | Result                                                                                                                                                                                                                                                    |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Independent forward 3D geometry and metadata contracts | 36 passed, including rotated and translated cameras, virtual rectification, differing grids, sentinels, invalid denominators, overflow, ties to even, padded rows, unchanged inputs, exact association, cache changes, passthrough, and double conversion |
| Actual BoxMeasurement reference equivalence            | 20 seeded depth fixtures and 1,000 independently reviewed camera-pose cases matched exactly                                                                                                                                                               |
| Relevant box, parcel, and correction regression run    | 116 passed, 1 optional dependency skip                                                                                                                                                                                                                    |
| Installed wheel outside checkout                       | 155 installed package and metadata files matched wheel bytes; isolated host pipeline passed                                                                                                                                                               |
| Host SDK fixture                                       | 3,331 valid pixels; maximum corrected source-plane residual 0.529797 mm; uncorrected residual 84.235476 mm                                                                                                                                                |
| OAK-4 standalone SDK fixture                           | Same valid count, residuals, geometry provenance, corrected depth bytes, and installed package digest as host                                                                                                                                             |
| Pre-commit                                             | All repository hooks passed on all tracked files, including new files                                                                                                                                                                                     |
| API documentation                                      | `pydoctor depthai_nodes` passed with warnings treated as errors                                                                                                                                                                                           |
| Frozen algorithm preservation                          | No changes to existing box nodes, parcel consumers, or bundled parcel implementation bytes                                                                                                                                                                |

The corrected depth SHA-256 is
`04c429146e2ff86aaa516f5eb547e467720470b44408ab1b1bcf0d99ff8f74be`.
The fixture uses a 96 by 64 source and 80 by 60 target, independent intrinsics,
source yaw 0.12 rad, pitch -0.08 rad, and translation 20, 3, 10 mm.
The corrected source-plane bound includes output millimeter quantization.
Source-Z behavior was asserted before correction on both execution paths.

The OAK-4 was an OAK4-D R9, serial `1631075878`, at `192.168.0.102`, running
Luxonis OS `1.47.0`, oak-agent `0.28.0`, and arm64 Python in the
`luxonis/oakapp-base:1.4.0-py311` app image. The controller used oakctl `0.32.0`.
ImageAlign and PointCloud used device placement. The correction ran on the
OAK-4 application CPU. No cameras, neural models, preview streams, Workbench
credentials, or workstation processing were used by the device fixture.
The one-shot app stopped after completion.

The full `pytest tests/unittests -q` command is not green on this checkout's
base. With the same exact SDK and environment, the untouched base had
575 passed, 50 failed, 93 errors, and 6 skipped. This branch had 577 passed,
50 failed, 93 errors, and 6 skipped. The 143 failing or errored test IDs are
identical. Existing mock-node import order and unavailable model-cache access
account for those failures. The focused regression run avoids that collection
order and passes. Raw baseline and branch logs are in the evidence bundle.

The behavioral reference was
`BoxMeasurement/boxlab/workbench/backend/native_rgbd.py`, read from checkout
revision `341e7585c853e551f2f90285a3f3cf9dfd54b734`. Its actual file SHA-256 was
`566a0df03f35c7a6a57daf6e6ae1ede46a95183cf479c21c5be374b747183be9`.
Independent review found that the SDK relative-pose convenience method uses
float32. Composing stored poses in float64 fixed a 637 mm input that otherwise
rounded to 570 mm instead of the reference's 569 mm. That case is a regression
test.

## Remaining acceptance work

The synthetic host and OAK-4 checks are complete. These broader benchmark and
production checks remain pending:

1. Verify real NeuralDepth 768 by 480 rectification metadata on retained input
   frames, including any live calibration changes.
1. Run an authorized snapshot from device X on OAK-4 Y with X's full applied
   geometry, native image associations, and support plane.
1. Compare frozen Workbench scenes for cloud count, order, validity, RGB decode,
   and unchanged parcel outcomes. Synthetic reference parity does not establish
   complete scene parity.
1. Adopt this exact wheel in BoxLabRunners and the designated production
   pipeline. Remove duplicate correction only after those equivalence checks.
   Keep invalid-depth filtering and the RGB optical-frame cloud contract at
   that boundary.

These tasks require the benchmark or production repositories and authorized
input bundles. This PR implements the shared component and its verification.
It does not publish a new PyPI release or claim neural-depth or algorithm
throughput.

## Implementation decisions

Geometry is an immutable tuple value. Its constructor validates and normalizes
inputs before construction, without explicit attribute mutation. The revised
artifact retains the original geometry digests and corrected depth bytes.

Model the Domain shaped the immutable geometry, explicit policy, and associated
provenance result. Prove It Works required checking installed wheel bytes and
executing actual host and standalone SDK pipelines. Sequence Work into
Verifiable Units separated the implementation commit from this evidence record.

The blocking work established the benchmark contract and exact SDK behavior.
One implementation owner kept correction policy and metadata consistent.
Independent geometry review and the pipeline, packaging, and hardware checks
used separate files. Git writes stayed with the primary agent. Both design
proposals converged on exact ImageAlign passthrough association. The final
design adds cloning with active-byte replacement to preserve padded layouts.

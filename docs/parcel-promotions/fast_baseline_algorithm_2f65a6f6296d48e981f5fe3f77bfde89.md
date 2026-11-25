# Fast baseline parcel promotion

This first parcel implementation binds `ParcelDetector` and `ParcelDimensioner` to `algorithm-2f65a6f6296d48e981f5fe3f77bfde89`.
There is no previous bundled parcel candidate.
Existing `BoxDetector`, `BoxDimensioner`, and `BoxMeasurer` implementations remain unchanged.
The original work is in [PR #348](https://github.com/luxonis/depthai-nodes/pull/348).
The packaging migration is prepared on `codex/parcel-stable-implementation`, based on PR revision `10f0b79d5473aada6face5d60a2e192c144d56ce`.
These changes are not a merged or published release.

## Frozen source and evidence

| Artifact                       | Identity                                                                  |
| ------------------------------ | ------------------------------------------------------------------------- |
| Algorithm and configuration    | `fast_baseline/v001`                                                      |
| Evaluated source               | `luxonis/BoxLabRunners@0a2f571843f81ac6cc2f8cebe3475540e314f83d`          |
| Committed evidence             | `luxonis/BoxLabRunners@4b7634cc5806b7fb2e48bc7a830a3434b85cc0cb`          |
| Executed image                 | `sha256:240c1854f46100c382801e8e0f622d99807c4ef64a0b27875f539087a97d7176` |
| Release                        | `release-f14a597a366c46c6bdba00fa25e45bcd`                                |
| Scoped settings byte SHA-256   | `66f29d1a54ddb6fd8d3fa957a34aa81dcef53a4ca29a47b0b4e39331afcb8bd8`        |
| Original native binary SHA-256 | `d777d52c77a31a2fede1f977edf28378e14504dc78a8d4479dc31054f0df2b31`        |

Before copying code, all 16 source-file hashes in the committed evidence matched the evaluated Git revision.
The scoped settings byte hash also matched. The retained image's native binary hash and runtime versions were checked directly.
`scoped-settings.json` is byte-identical to the committed release-scoped settings at the evidence revision above.
BoxLabRunners prepared that artifact by omitting an inaccessible historical parent-registration declaration from earlier registration settings.
The port makes no further edit. The aggregate evaluation evidence also retains its committed bytes.
The settings preserve computation, policy, objective, and input declarations.

The revision design maps these settings to `fast_baseline/v001`, with canonical path `algorithms/fast_baseline/configurations/v001.json`.
That canonical copy has the same bytes. It does not retroactively change the evaluated source tree, image, mounted settings path, or journal.
The mapping is committed in [BoxLabRunners PR #15](https://github.com/luxonis/BoxLabRunners/pull/15) at `df216e5253c6f29de677b83c4ad7248cb020f871`. That PR remains open; this mapping does not claim a newly reviewed or evaluated algorithm.
The manifest separately pins its [configuration catalog](https://github.com/luxonis/BoxLabRunners/blob/df216e5253c6f29de677b83c4ad7248cb020f871/experiments/configurations.json) and [evaluation record](https://github.com/luxonis/BoxLabRunners/blob/df216e5253c6f29de677b83c4ad7248cb020f871/experiments/fast_baseline/v001/release-f14a597a366c46c6bdba00fa25e45bcd/2026-11-20-revalidation/record.json), including their file hashes.

The [manifest](../../depthai_nodes/algorithms/parcel/manifest.json) separates original candidate hashes from the port's raw file and configuration hashes.
Its complete validation and approved held-out evaluation records preserve `boxlab-run-metrics/3`, scoring policy, run IDs, denominators, null reasons, and suppression.
Exact held-out dimensional medians and p95 errors remain null and suppressed.
The historical latency reference belongs to `algorithm-8bdaf89d28cf45f29712cb68e1b31357` and its original image.
The original port verification did not run a timing benchmark.
The accepted historical means of 116.32 ms on validation and 105.33 ms on held-out scenes are not live OAK latency claims.
Later OAK-4 diagnostics and outstanding algorithm findings are recorded separately in [upstream-findings.md](upstream-findings.md).

## Source adaptations

The single stable package is `depthai_nodes/algorithms/parcel/implementation/`.
Git history and retained package releases preserve earlier implementation revisions.
This record retains the full candidate identity independently of the directory name.

| Frozen source                               | Port adaptation                                                                                                                       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `algorithms/fast_baseline/__init__.py`      | Relocated to `factory.py`; lightweight package imports defer SciPy until use.                                                         |
| `geometry.py`, `visibility.py`, `native.py` | Replace Runner imports with bundled neutral types; apply repository formatting.                                                       |
| `kernels.cpp`, `build_native.py`            | Preserve native computation and exact compiler flags. The C++ source is byte-identical.                                               |
| `runner/scene.py`                           | Retain the computation dataclasses, `SceneError`, and `project_points`; omit artifact transport and decoding interfaces from runtime. |
| `runner/components.py`                      | Retain the two callback fields. Omit the unused plane-estimator field and Workbench result transport.                                 |
| Scoped settings                             | Preserve original bytes and serialize their unchanged `configuration` object separately.                                              |

Computation ASTs agree after removing imports and docstrings. `project_points` retains the original precision and distortion handling.
No threshold, sampling, fallback, point order, candidate order, or visibility policy changed.
Native workers complete before a scene callback returns or raises. Per-scene caches reset on repeated calls.
The source attribution records Luxonis authorship and the absence of a separate upstream license file at the pinned revision.

The portable wheel contains no native binary. It includes C++17 sources and the original NumPy/OpenCV fallback.
An optional local build uses `-std=c++17 -O3 -ffp-contract=off -fPIC -shared`.
The retained Linux aarch64 binary is never presented as portable.

## Public API and frame contract

```python
ParcelDetector.build(pointCloud, color, left, right, supportPlane=None)
ParcelDimensioner.build(parcels)
```

Both builders return their node. Both expose `out` and bind the same internal no-argument component factory.
Neither exposes algorithm selection, injected backends, frozen settings, or threshold overrides.
Install `depthai-nodes[parcel]` for computation.

`pointCloud` produces `dai.PointCloudData`. The three images produce native `dai.ImgFrame` messages.
The cloud contains finite float32 XYZ in millimeters in the explicitly declared scene frame.
The detector preserves every point and its order. It rejects invalid input rather than filtering points silently.
Color accepts NV12 and native 8-bit RGB/BGR layouts and becomes RGB. Left and right accept GRAY8 or RAW8.
The detector performs no resize. Intrinsics and distortion describe each delivered pixel grid.
Transforms map scene coordinates into each camera's optical frame, with translations in millimeters.

Use the evaluated unrectified source-image profile:

| Input | Width × height | Geometry    |
| ----- | -------------- | ----------- |
| RGB   | 2000 × 1500    | Unrectified |
| Left  | 1280 × 800     | Unrectified |
| Right | 1280 × 800     | Unrectified |

The first configured scene with valid message types checks all three image sizes. A mismatch logs one warning containing all expected and received sizes.
This check runs once per detector instance, including when the initial sizes match. Invalid message types do not consume the check.
Processing continues after a mismatch.
NeuralDepth at 768 × 480 is preferred upstream. The detector receives a point cloud and cannot check the depth-image resolution directly.
The synthetic example deliberately uses smaller images and emits this warning. It verifies pipeline wiring and geometry, not the evaluated input profile.

Camera calibration is read from each `ImgFrame.getTransformation()`:

```python
metadata.getIntrinsicMatrix()
metadata.getDistortionModel()
metadata.getDistortionCoefficients()
metadata.getExtrinsics().getTransformationMatrix(False, dai.LengthUnit.MILLIMETER)
```

`getIntrinsicMatrix()` includes delivered crop/scale transforms. The original sensor's `getSourceIntrinsicMatrix()` is not a substitute.
Camera poses are inverted with NumPy in float64 to map common-reference XYZ into each camera.
All three images and the cloud must declare a compatible target device and camera socket.
Missing or invalid metadata, inconsistent grids, incompatible reference frames, and unsupported distortion models produce failed frames.
The adapter creates no placeholder image or replacement calibration.

Native DEFAULT point clouds with an identity custom transform already contain XYZ in the shared calibration reference frame.
Their image metadata can retain the depth camera's source pose. Applying that pose to the cloud again would be incorrect.
The adapter uses each image's reference-to-camera transform directly, without applying the cloud's source pose twice.
This follows the [tagged native coordinate transformation](https://github.com/luxonis/depthai-core/blob/v3.11.0/src/pipeline/node/PointCloud.cpp#L626) and [metadata copying](https://github.com/luxonis/depthai-core/blob/v3.11.0/src/pipeline/node/PointCloud.cpp#L843).
Custom transforms, housing coordinates, and camera-target cloud modes are outside this adapter's contract because the message lacks a reliable frame-mode discriminator.
The builder rejects incompatible initial native PointCloud configurations. Coordinate mode, units, and custom transforms must remain fixed while the pipeline runs.
Other producers must already provide millimeter XYZ in the declared common reference frame with consistent metadata.

The optional fifth builder argument is an existing `SupportPlane` value in the common point-cloud reference frame.
If omitted or `None`, the detector consumes synchronized groups without decoding, detection, or outputs until a `ParcelDetectorConfig` supplies a plane.
Those groups are discarded, not retained for later processing. Capture and configuration polling continue without a pipeline restart.
The reference camera must be above the support surface. The detector orients either equation sign toward that camera, producing a positive offset.
A plane through the reference origin cannot be oriented this way and produces a failed frame.
The batch frame label is derived from the shared native target device and camera socket.

`ParcelDetectorConfig` is a same-host `dai.Buffer` dataclass with exactly one field, `supportPlane`.
Send it through `detector.inputConfig.createInputQueue(blocking=True)` to update the runtime plane. There is no `setSupportPlane()` method or separate plane-stream message.
One queued configuration is applied before each synchronized group, when available. Otherwise, the stored configuration remains active.
Updates are applied in queue order, one per scene. An update arriving during processing applies to a later group.
Config input messages with an incorrect datatype log a warning and leave the active state intact.
Each batch retains the exact oriented plane used for its scene; queued batches cannot observe subsequent updates in the dimensioner.

`ParcelDetector` is a `dai.node.ThreadedHostNode` with a native host `Sync` subnode for `pointCloud`, `color`, `left`, and `right`.
`run()` blocks on the synchronized scene queue, then reads one pending config message with `tryGet()`.
When an update is available, `_update_config()` applies it to stored state. `process()` receives only the four scene messages and uses the active configuration.
The config queue is blocking and bounded.
The combined native pipeline crashed on DepthAI 3.11 with a one-entry nonblocking config queue and two queued Python config messages.
The blocking queue passes that same case with updates applied in queue order. A standalone queue probe did not reproduce the crash, so its broader SDK cause is not established.
If scenes stop and the config queue fills, config producers experience backpressure until another scene is processed.

Scene synchronization uses DEVICE timestamps and a span below 10 ms. All four streams must share a device clock.
Sequence counters do not need to match; the output preserves the point cloud's counter and each source's individual metadata.
An incomplete group produces no callback or empty success. The next complete group can recover. There is no custom scene-matching loop or timeout.
Specific scene datatypes are checked once in public `process()`. Private conversion helpers trust those types and validate content such as calibration and finite XYZ.

## Image-to-cloud assumptions

The point cloud is a 3D point set. It does not need pixel-to-row correspondence with any image and need not be organized.
The images need not be the exact buffers used by the point-cloud generator. They must show the same scene during the synchronized capture interval.
Each image must have valid calibration into the shared point-cloud reference frame. Preserve the evaluated native field of view, orientation, and unrectified geometry.
The algorithm projects XYZ into each view; it never resizes images to match the point-cloud grid.
Its absolute pixel thresholds make eligibility resolution-dependent. Other resolutions, aspect ratios, and crops may change decisions even with correct calibration.

The adapter can decode raw and rectified views when metadata describes the actual delivered pixels, but rectified inputs fall outside the evaluated profile.
Raw images retain their lens distortion. Rectified images require the virtual-camera intrinsics and pose, with distortion removed by the rectifier.
Matching dimensions alone do not prove matching geometry. The resolution warning does not classify rectification or detect every crop.
The adapter must not pair rectified pixels with the original raw-camera calibration.
Native host `Rectification` was tested with distorted 64 by 48 inputs and 80 by 36 RAW8 outputs; its metadata had updated intrinsics, rectified poses, and empty distortion.
Native StereoDepth rectified outputs are declared RAW8, but their device-firmware metadata has not been tested on hardware here.

## Parcel batch and measurement

`ParcelDetections` contains eligible `Parcel` items with a zero-based frame-local `candidate_id` and owned read-only `xyz_mm`.
Rejected proposals can leave gaps in IDs. `decisions` retains proposal outcomes without source-point indices or the whole cloud.
Each parcel is extracted once with `numpy.take` in selected-point order. The dimensioner retains that array without another extraction or copy.
Parcel value objects use ordinary constructors and read-only properties, without explicit `object.__setattr__` calls.
The public parcel constructor snapshots all inputs into immutable bytes-backed float32 storage.
The internal owned-array path also snapshots the extracted points. This adds one boundary copy after `numpy.take`.
Retained writable aliases cannot modify the published points, and callers cannot re-enable writes through `xyz_mm`.
The dimensioner adds no point copy. Point values, shape, and selected-point order are unchanged.
`sources` contains timing records with `stream`, `timestamp`, `timestamp_device`, and `sequence_num` fields for every available input.
The source-timing and diagnostic-decision record classes are internal. Native output timestamps and sequence repeat the source cloud.

`ParcelMeasurements` contains `ParcelMeasurement` items. `ParcelMeasurement.Status` distinguishes `NO_FIT`, `HEIGHT_UNKNOWN`, and `MEASURED`.
A fitted `ParcelCuboid` has `base_center_mm`, `length_axis`, `length_mm`, `width_mm`, and optional `height_mm`.
The center lies on the carried plane. The unit length axis is tangent to it. The width axis is the upward plane normal crossed with the length axis.
Both batch messages expose the same `Outcome` enum, such as `ParcelDetections.Outcome.SUCCESS` and `ParcelMeasurements.Outcome.FAILED`.
A successful empty scene remains successful through both nodes.
Input failure, identity mismatch, or callback failure produces a failed frame with no partial results.
The dimensioner reads only the batch's plane. No tracking, temporal medians, or confidence is added.

These Python messages connect nodes on one host. Nested arrays have no implemented XLink or remote serialization.
The [synthetic example](../../examples/parcel_measurement/synthetic.py) runs both nodes in a native host-only DepthAI pipeline without an OAK device.

The message package exports six parcel types. `ParcelDetectorConfig` carries support-plane updates.
`Parcel`, `ParcelDetections`, `ParcelCuboid`, `ParcelMeasurement`, and `ParcelMeasurements` describe public payloads consumed downstream.
Status enums live under their owning types. Internal timing and diagnostic records add no flat public constructors.

## Dependency resolution

The original image uses Python 3.12.16 on Linux aarch64 with NumPy 2.2.6, SciPy 1.15.3, and `opencv-python-headless==4.12.0.88`.
The target base originally constrained OpenCV to `~=4.10.0`.
Source-versus-port comparisons pass exactly under both OpenCV 4.10.0.84 and 4.12.0.88 in separate macOS arm64 environments.
Cross-version comparisons preserve candidate order, membership, point bytes, decisions, and fit outcomes.
Two validation cuboids differ in geometry. The maximum dimension/center difference is `8.922520180476567e-6` mm.
The maximum length-axis component difference is `6.107310285274892e-8`.
These differences exceed the declared `1e-9` absolute parity tolerance. They are not treated as equality.

The base requirement now permits `opencv-python-headless>=4.10.0,<4.13.0`.
The optional parcel extra pins OpenCV to `4.12.0.88`, NumPy to `2.2.6`, and SciPy to `1.15.3`.
This resolves the incompatible original pins and keeps the selected computation dependencies explicit.
Existing box regressions were checked with OpenCV 4.10 and 4.12. The parcel nodes remain importable without SciPy.
Installing with the declared parcel extra selects the frozen dependencies. A manually overridden dependency environment is outside that verified binding.

## Original comparison coverage

The tolerance was declared before comparison. Indices, candidate order, decisions, parcel points, and point order must agree exactly.
Cuboid values use absolute tolerance `1e-9` and relative tolerance zero. Fit and unknown-height outcomes must agree exactly.

The original comparison ran in both dependency environments on 15 synthetic scenes and all 18 scenes in the authorized historical validation cache.
Both native and fallback modes run every scene. Source-versus-port geometry differences are zero in each environment.
The same-environment native and fallback results were also identical on the compared scenes.
This is limited coverage. A later explicit cell-boundary case reproduces a native/fallback discrepancy inherited from the evaluated source, as detailed in [upstream-findings.md](upstream-findings.md).
Separate measurement cases cover a tilted parcel with outliers, unknown height, and no usable footprint.

The cache manifest SHA-256 is `52fb068a961ffb9e7303d3bd8abab20df13ff78cc71e35997c7bad17886b8c84`.
The comparison tool verifies the canonical cache manifest and every cached scene hash before decoding arrays.
This historical cache is from the same release. Its manifest differs from the selected revalidation run's `baec14e249caa505bf16a86abdf4ea31e888f631eda0e955d641406ac53c5535`.
These checks establish port equality on the named decoded scenes. They do not rerun scoring, establish equality on new held-out data, or verify original raw recording decoding.
No training labels or new held-out inputs were accessed.

The [numerical comparison report](comparison-summary.json) contains aggregate checks and hashes, without raw scenes or validation cuboid coordinates.
The original retained image separately passes 25 fast-baseline regressions and seven component checks without skips.
The port reruns the adapted regressions and checks its actual native message boundary separately.

To repeat a source comparison after obtaining the exact pinned reference tree and already-authorized cache:

```sh
python tools/verify_parcel_port.py \
  --reference-root /path/to/verified-reference \
  --validation-cache /path/to/authorized-validation-cache \
  --require-native --output /tmp/parcel-comparison.json
```

The reference root contains the pinned source plus committed `fast-baseline.json` and `fast-baseline-settings.json`.
Build both local native libraries before using `--require-native`. Omit that option for fallback-only coverage when native builds are unavailable.
No fetching or network access occurs in this tool.

## Original repository checks and limitations

The results below describe the earlier port revisions and their recorded artifacts.
They are retained as historical verification and do not identify a newly built stable-directory wheel.

The promotion checks ran on macOS arm64 with the declared parcel dependencies: NumPy 2.2.6, SciPy 1.15.3, and OpenCV package 4.12.0.88.
Source checks used Python 3.10.18 and 3.12.13 with DepthAI 3.11.0. The installed-wheel checks used Python 3.12.13 with DepthAI 3.12.2.
The numerical source comparisons and OpenCV compatibility checks are recorded separately in [comparison-summary.json](comparison-summary.json).

| Check                                                                                                   | Recorded result                                                                                                                      |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `pytest tests/unittests/test_parcel_algorithm tests/unittests/test_parcel_nodes tests/parcel_reference` | 42 tests pass on both Python versions, including frozen regression subtests and 49 native boundary checks in an isolated subprocess. |
| `pytest tests/unittests`                                                                                | 581 pass, 181 fail, and 124 errors. The unchanged base reproduces the same failures and errors, with 571 passes.                     |
| Isolated host-node tests                                                                                | 116 pass with OpenCV 4.10 and 4.12.                                                                                                  |
| Helpers and message tests                                                                               | 382 pass with OpenCV 4.10. Combined host, helper, and message checks pass 498 tests with OpenCV 4.12.                                |
| Focused existing box threaded-node tests                                                                | 31 pass with OpenCV 4.10 and 4.12.                                                                                                   |
| `pytest tests/unittests --ignore=tests/unittests/test_nodes`                                            | 524 pass.                                                                                                                            |
| Separate threaded-node suite                                                                            | 195 pass and 16 fail because model caches are unavailable. The unchanged base reproduces those failures.                             |
| `pre-commit run --all-files`                                                                            | Passes.                                                                                                                              |
| `pydoctor depthai_nodes`                                                                                | Passes with pydoctor 25.10.1 and warnings treated as errors.                                                                         |

The full-suite failures include an existing collection-order problem in parser-node mocks. Native parcel checks run in a fresh process to avoid those mock substitutions.
The native checks cover:

- Incomplete synchronized groups, recovery on the next complete group, and independent sequence counters.
- Config updates in queue order, continued processing without new configs, updates during processing, invalid-config warnings, and retained batch plane snapshots.
- Optional initial planes, consumed groups without decoding or outputs while unconfigured, and activation from a later config without replaying discarded groups or restarting the pipeline.
- Expected image resolutions without warnings, one warning for each mismatched role across repeated scenes, continued processing, and invalid message types leaving the check pending.
- Crop/scale intrinsics, distortion, tilted sensors, pose units, reference compatibility, missing metadata, and default cloud coordinates.
- Read-only parcel values, the earlier owned-array transfer, measurement outcomes, and algorithm identity checks. The migration strengthens storage immutability as described above.

Native host Rectification from distorted 64×48 inputs to 80×36 RAW8 outputs passes metadata-conversion checks.
This does not validate rectified inputs against the evaluated unrectified profile or establish generated-cloud or live camera behavior on OAK hardware.

### Original installed package

The wheel installs outside both source trees. All 24 covered installed file hashes and the declared JSON, configuration, attribution, and C++ assets pass validation.
The package excludes test fixtures, verification tools, documentation packages, and native binaries.
Before installing SciPy, public imports, six parcel message exports, the single-field config dataclass, and manifest validation pass.
Creating computation without the parcel extra produces the installation hint.

After installing the parcel extra, all 49 native boundary checks pass from the installed package.
A host-only synthetic pipeline runs both public nodes with the fallback and an optional native build using Apple clang 21 and the declared C++17 flags.
Both paths emit the expected one-time resolution warning for the small synthetic images and measure the 220 by 160 by 110 mm parcel as 218.940 by 158.420 by 110.000 mm.
The fixture allows 5 mm for sampling. That allowance does not relax the source-to-port numerical tolerance.
The final reinstall verifies the input-profile provenance, all installed file hashes, and the fallback pipeline.

The tested input-profile revision wheel has SHA-256 `f60ef2ae9ef4eed34c46916568176fe7c6dffffc4cf8ffc5dbc8d85b9012b578`.
This identifies the previously checked artifact, not a wheel rebuilt after later documentation or provenance changes.
Native binary identities are retained under `implementation.tested_native_binaries` in the [manifest](../../depthai_nodes/algorithms/parcel/manifest.json).

For the installed-wheel check, use a new virtual environment in a directory outside both repositories:

```sh
python -m pip install /path/to/depthai_nodes-0.8.0-py3-none-any.whl
python -c 'from depthai_nodes.node import ParcelDetector, ParcelDimensioner; from depthai_nodes.algorithms.parcel import validate_manifest; validate_manifest()'
python -m pip install '/path/to/depthai_nodes-0.8.0-py3-none-any.whl[parcel]'
python /path/to/copied/synthetic.py
python -m depthai_nodes.algorithms.parcel.implementation.build_native
python /path/to/copied/synthetic.py
```

The test-only report comparison can be repeated with `python -m tests.parcel_reference.compare_reports left.json right.json --output comparison.json`.
It returns a failing equality result for the observed OpenCV 4.10/4.12 drift at the unchanged tolerance.

The original verification did not test OAK hardware, installed Linux or Windows port execution, XLink serialization, or an end-to-end live sensor pipeline.
The subsequent OAK-4 investigation is separate diagnostic evidence in [upstream-findings.md](upstream-findings.md); its one-scene prototypes have not been promoted.
The portable fallback and optional native build have distinct verification records. No hardware performance guarantee is made.

## Stable-directory migration

Both nodes keep their public APIs and bind through the same private factory to `fast_baseline/v001` and the full candidate identity above.
Imports, package-data declarations, tests, native-build instructions, and manifest paths now use `implementation/`.
The configuration and computation remain frozen. The pre-migration PR contains no algorithm correction relative to the evaluated source.
The audit checked computation ASTs after import and docstring adaptations and exact C++ and scoped-settings bytes.
Directory relocation does not establish evaluated-artifact identity; published port hashes remain separate.

The existing `visualization.html` is now a wheel asset covered by the manifest.
Its source documentation revision is `1d2aa20e85f6f6b6acb597440ca42d85c495474f`, later than the evaluated computation revision.
Its source SHA-256 is `aa8837d8b0eaa28261986c5f56681bce84431e77cbd365413b5debf470ff8c71`.
Source links are repaired for shipped files or commit-pinned URLs. The later documentation UI correction restores keyboard focus to the same record and axis after chart redraw. Embedded candidate identity, configuration, metrics, and case data stay attributed to their original evidence.
The manifest records the destination hash separately.

The migration strengthens external-reference verification by checking it against repository-owned evidence before importing reference code.
It also strengthens the worker-completion test and parcel storage immutability, and documents a blocking config producer queue.
These are integration and verification changes. The [upstream findings](upstream-findings.md) classify algorithm corrections and the stale-native-library ownership question.

A new source-versus-port comparison runs 15 synthetic scenes in native and fallback modes, plus standalone tilted-fit, no-fit, and unknown-height cases.
The historical 18-scene validation cache was unavailable for this rerun. Its earlier numerical report remains [comparison-summary.json](comparison-summary.json).
No new Workbench scoring or held-out evaluation ran. The current candidate identity and historical evidence remain unchanged.

The migration checks pass 53 focused tests and 58 subtests on Python 3.10.18 and 3.12.13, including 50 native boundary checks.
The 498 combined helper, message, and host tests and 31 focused threaded box tests pass.
The API reference builds with warnings treated as errors.
All 13 relocated Python, C++, JSON, and dependency-text files match the PR baseline bytes exactly.
The complete pre-relocation and post-relocation comparison reports match for all 15 synthetic scenes and three standalone cases in native and fallback modes.
The full unit suite reports 592 passes, 181 failures, and 124 errors.
The unchanged PR revision reports 581 passes with the same 181 failures and 124 errors.
All 305 failing or erroring test IDs match. The focused suites above pass in isolation.

The wheel was built from a fresh staging tree because the existing build directory retained deleted candidate-named packages.
The clean wheel contains only the stable implementation, all 25 covered files, and the visualization. It contains no native binary.
Installed imports, manifest hashes, candidate/configuration binding, optional SciPy import handling, and all 50 native message checks pass outside both repositories on Python 3.12.13 and DepthAI 3.12.2.
Fallback and a fresh local native build both measure the synthetic parcel as 218.940 by 158.420 by 110.000 mm.
The local binary hash is recorded separately in the manifest and is not a distributed artifact.

The initial verified stable-directory wheel has SHA-256 `65d31a87c09429ba696d5b42b4eb45cffdd91cfadaacf95f083e27a651119b60`.
The wheel contains 156 files and its manifest bytes match the source manifest.
This identifies the tested build, not future rebuilds or a published release.
After the BoxLabRunners migration commit became available, its commit-pinned catalog and evaluation-record references were added without changing any covered port file.
The final wheel with those references has SHA-256 `c661200716e15d798de84d601eda1a2c3a671adee9c1ebd0bafae4255c92f379`; its 25 covered files and installed manifest validate again.

The fresh test environment used previously installed pinned dependencies after new dependency downloads stalled.
Only the new depthai-nodes wheel was installed as the package under test. No depthai-nodes checkout files or import paths were copied into that environment.
Its dependency versions are NumPy 2.2.6, SciPy 1.15.3, and OpenCV 4.12.0.88.
Pre-commit and the warnings-as-errors API reference build pass.

Future algorithm corrections require BoxLabRunners review and evaluation before this repository integrates their new frozen revision and evidence.
The canonical configuration-path migration does not require re-evaluating these unchanged settings, and its committed catalog and evaluation-record provenance are now recorded separately from the evaluated source.
Future changes follow the [update workflow](update-workflow.md), also linked from `CONTRIBUTING.md`.
Rollback restores the complete prior code, settings, dependencies, assets, and manifest release.

## Review corrections on November 24, 2026

The validator and affected tests now use explicit UTF-8 for text assets, with a regression exercising a cp1252 default. The preceding Windows CI run failed with a decode error at byte 21095; the updated Windows results are retained in PR #348 checks.

The visualization restores focus after Enter and Space activate a chart mark. Both embedded evidence datasets remain unchanged. The HTML and validator have refreshed production-port hashes; the evaluated-source identity and original visualization hash are retained. This is a packaging/documentation correction, not a newly evaluated algorithm. Earlier wheel hashes above continue to identify their exact prior artifacts.

The captured-scene prototype report and handoff now limit hash agreement to the checked adapter output. The hash does not establish proposal indices, point membership/order, or per-proposal decision parity. Raw retained records contain output hashes and separate decision counts. Stronger prototype parity remains a separate BoxLabRunners task.

The upstream allocation backlog now includes both `centered_hull_candidates` and `group_rows`. Their inherited code remains unchanged here; corrections require upstream review and evaluation.

The corrected wheel has SHA-256 `b1ba20a7195a034e9cebf971b93677d03d5c05e4e910558efca2bd5a882dfa3f`. All 25 installed hashes validate with cp1252 as the simulated locale default. The installed page passes 12 real-browser focus checks covering Enter and Space, all three axes, and signed/absolute mode. The focused suite passes 54 tests and 58 subtests on Python 3.10 and 3.12.

## Retained native-build evidence

The predecessor C++17 build flags and all six local binary records are retained verbatim in [historical native-build provenance](historical-native-builds-algorithm-2f65a6f6296d48e981f5fe3f77bfde89.json), pinned to the predecessor manifest revision. They do not apply to the replacement visibility v002 implementation.

# Fast baseline parcel promotion

This first parcel implementation binds `ParcelDetector` and `ParcelDimensioner` to `algorithm-2f65a6f6296d48e981f5fe3f77bfde89`.
There is no previous bundled parcel candidate.
Existing `BoxDetector`, `BoxDimensioner`, and `BoxMeasurer` implementations remain unchanged.
The work is prepared on `codex/parcel-fast-baseline`. It is not merged or published.

## Frozen source and evidence

| Artifact                       | Identity                                                                  |
| ------------------------------ | ------------------------------------------------------------------------- |
| Evaluated source               | `luxonis/BoxLabRunners@0a2f571843f81ac6cc2f8cebe3475540e314f83d`          |
| Committed evidence             | `luxonis/BoxLabRunners@4b7634cc5806b7fb2e48bc7a830a3434b85cc0cb`          |
| Executed image                 | `sha256:240c1854f46100c382801e8e0f622d99807c4ef64a0b27875f539087a97d7176` |
| Release                        | `release-f14a597a366c46c6bdba00fa25e45bcd`                                |
| Scoped settings byte SHA-256   | `66f29d1a54ddb6fd8d3fa957a34aa81dcef53a4ca29a47b0b4e39331afcb8bd8`        |
| Original native binary SHA-256 | `d777d52c77a31a2fede1f977edf28378e14504dc78a8d4479dc31054f0df2b31`        |

Before copying code, all 16 source-file hashes in the committed evidence matched the evaluated Git revision.
The scoped settings byte hash also matched. The retained image's native binary hash and runtime versions were checked directly.
The scoped settings omit only an inaccessible historical parent-registration declaration.
They preserve computation, policy, objective, and input declarations. The packaged evidence preserves that lineage explanation.

The [manifest](../../depthai_nodes/algorithms/parcel/manifest.json) separates original candidate hashes from the port's raw file and configuration hashes.
Its complete validation and approved held-out evaluation records preserve `boxlab-run-metrics/3`, scoring policy, run IDs, denominators, null reasons, and suppression.
Exact held-out dimensional medians and p95 errors remain null and suppressed.
The historical latency reference belongs to `algorithm-8bdaf89d28cf45f29712cb68e1b31357` and its original image.
No timing benchmark ran for this port. The accepted historical means of 116.32 ms on validation and 105.33 ms on held-out scenes are not live OAK latency claims.

## Source adaptations

The immutable package is `depthai_nodes/algorithms/parcel/fast_baseline_algorithm_2f65a6f6296d48e981f5fe3f77bfde89/`.

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
ParcelDetector.build(pointCloud, color, left, right, supportPlane)
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

The fifth builder argument is the existing `SupportPlane` value, in the common point-cloud reference frame.
The reference camera must be above the support surface. The detector orients either equation sign toward that camera, producing a positive offset.
A plane through the reference origin cannot be oriented this way and produces a failed frame.
The batch frame label is derived from the shared native target device and camera socket.

`ParcelDetectorConfig` is a same-host `dai.Buffer` dataclass with exactly one field, `supportPlane`.
Send it through `detector.inputConfig.createInputQueue()` to update the runtime plane. There is no `setSupportPlane()` method or separate plane-stream message.
The latest queued full configuration applies before the next synchronized group is processed. An update arriving during processing applies to a later group.
Each batch retains the exact oriented plane used for its scene; queued batches cannot observe subsequent updates in the dimensioner.

`ParcelDetector` is a `dai.node.ThreadedHostNode` with a native host `Sync` subnode for `pointCloud`, `color`, `left`, and `right`.
`run()` blocks on the synchronized scene queue, then reads pending config messages with `tryGet()` before processing.
The config queue is blocking and bounded. Available updates are drained in Python to retain the latest full state.
The combined native pipeline crashed on DepthAI 3.11 with a one-entry nonblocking config queue and two queued Python config messages.
The blocking queue with Python draining passed that same case. A standalone queue probe did not reproduce the crash, so its broader SDK cause is not established.
If scenes stop and the config queue fills, config producers experience backpressure until another scene is processed.

Scene synchronization uses DEVICE timestamps and a span below 10 ms. All four streams must share a device clock.
Sequence counters do not need to match; the output preserves the point cloud's counter and each source's individual metadata.
An incomplete group produces no callback or empty success. The next complete group can recover. There is no custom scene-matching loop or timeout.
Specific scene datatypes are checked once in public `process()`. Private conversion helpers trust those types and validate content such as calibration and finite XYZ.

## Image-to-cloud assumptions

The point cloud is a 3D point set. It does not need pixel-to-row correspondence with any image and need not be organized.
The images need not be the exact buffers used by the point-cloud generator. They must show the same scene during the synchronized capture interval.
Each image must have valid calibration into the shared point-cloud reference frame. Independent resolutions, aspect ratios, crops, and fields of view are supported.
The algorithm projects XYZ into each view; it never resizes images to match the point-cloud grid. Cropping may make a parcel ineligible under the frozen visibility policy.

Raw and rectified views are supported when metadata describes the actual delivered pixels.
Raw images retain their lens distortion. Rectified images require the virtual-camera intrinsics and pose, with distortion removed by the rectifier.
The adapter must not pair rectified pixels with the original raw-camera calibration.
Native host `Rectification` was tested with distorted 64 by 48 inputs and 80 by 36 RAW8 outputs; its metadata had updated intrinsics, rectified poses, and empty distortion.
Native StereoDepth rectified outputs are declared RAW8, but their device-firmware metadata has not been tested on hardware here.

## Parcel batch and measurement

`ParcelDetections` contains eligible `Parcel` items with a zero-based frame-local `candidate_id` and owned read-only `xyz_mm`.
Rejected proposals can leave gaps in IDs. `decisions` retains proposal outcomes without source-point indices or the whole cloud.
Each parcel is extracted once with `numpy.take` in selected-point order. The dimensioner retains that array without another extraction or copy.
Parcel value objects use ordinary constructors and read-only properties, without explicit `object.__setattr__` calls.
The public parcel constructor retains an owning read-only float32 array and snapshots writable or non-owning inputs.
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

## Verification scope

The tolerance was declared before comparison. Indices, candidate order, decisions, parcel points, and point order must agree exactly.
Cuboid values use absolute tolerance `1e-9` and relative tolerance zero. Fit and unknown-height outcomes must agree exactly.

Both dependency environments compare 15 synthetic scenes and all 18 scenes in the authorized historical validation cache.
Both native and fallback modes run every scene. Source-versus-port geometry differences are zero in each environment.
The same-environment native and fallback results are also identical on the compared scenes.
Separate measurement cases cover a tilted parcel with outliers, unknown height, and no usable footprint.

The cache manifest SHA-256 is `52fb068a961ffb9e7303d3bd8abab20df13ff78cc71e35997c7bad17886b8c84`.
The comparison tool verifies the canonical cache manifest and every cached scene hash before decoding arrays.
This historical cache is from the same release. Its manifest differs from the selected revalidation run's `baec14e249caa505bf16a86abdf4ea31e888f631eda0e955d641406ac53c5535`.
These checks establish port equality on the named decoded scenes. They do not rerun scoring, establish equality on new held-out data, or verify original raw recording decoding.
No training labels or new held-out inputs were accessed.

The committed comparison summaries contain aggregate checks and hashes, without raw scenes or validation cuboid coordinates.
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

## Repository checks and limitations

The exact full `pytest tests/unittests` command encounters an existing collection-order problem in parser-node mocks.
An untouched copy of the original revision reproduces the same 181 failures and 124 errors, with 571 passing tests.
The isolated host-node suite passes 116 tests. Helpers/messages pass 382 tests. Focused existing box threaded-node regressions pass 31 tests.
The separate threaded-node suite reproduces the base's 16 unavailable model-cache failures, with 195 tests passing.
These unrelated failures are retained as limitations rather than hidden as skips.

Final parcel, Python 3.10, wheel, lint, and documentation results are recorded in the manifest and comparison summaries.
The focused command `pytest tests/unittests/test_parcel_algorithm tests/unittests/test_parcel_nodes tests/parcel_reference` passes 42 tests, including frozen regression subtests, on both Python 3.12.13 and 3.10.18.
The node wrapper runs 44 native checks in a fresh process to avoid the existing parser mock substitutions.
The native synchronization cases verify that incomplete input groups are skipped, the next complete frame recovers, and queued plane configurations apply between complete scenes.
Native metadata checks cover crop/scale intrinsics, distortion and tilted sensors, pose units and reference compatibility, missing metadata, default cloud coordinates, normal value construction, and zero-copy owned parcel transfer.
These checks include native host Rectification from distorted 64×48 inputs to 80×36 RAW8 outputs with matching virtual metadata. They do not establish generated-cloud or live camera behavior on OAK hardware.
The final full unit command reports 581 passing tests with the same 181 failures and 124 errors as the unchanged base.
Pre-commit and `pydoctor depthai_nodes` pass. The API build treats warnings as errors.

The wheel installs outside both source trees. Its metadata pins the parcel dependencies, its 24 covered installed file hashes pass, and its JSON, configuration, attribution, and C++ assets are present.
The wheel excludes test fixtures, verification tools, documentation packages, and local native binaries.
Public package/node imports pass before installing SciPy. Creating computation without the extra gives the installation hint.
After installing the declared parcel extra, a host-only synthetic pipeline runs both public nodes from the installed wheel with the fallback.
The optional native source also builds inside that installation, and the same pipeline passes with it loaded.
The fresh wheel resolves DepthAI 3.12.2 and passes all 44 native boundary checks with that SDK. Source-tree checks use DepthAI 3.11.0.
Both paths measure the synthetic 220 by 160 by 110 mm parcel as 218.940 by 158.420 by 110.000 mm, within the fixture's 5 mm sampling allowance.
That allowance checks synthetic scene sampling. It does not relax the source-to-port numerical tolerance.

For the installed-wheel check, use a new virtual environment in a directory outside both repositories:

```sh
python -m pip install /path/to/depthai_nodes-0.8.0-py3-none-any.whl
python -c 'from depthai_nodes.node import ParcelDetector, ParcelDimensioner; from depthai_nodes.algorithms.parcel import validate_manifest; validate_manifest()'
python -m pip install '/path/to/depthai_nodes-0.8.0-py3-none-any.whl[parcel]'
python /path/to/copied/synthetic.py
python -m depthai_nodes.algorithms.parcel.fast_baseline_algorithm_2f65a6f6296d48e981f5fe3f77bfde89.build_native
python /path/to/copied/synthetic.py
```

The test-only report comparison can be repeated with `python -m tests.parcel_reference.compare_reports left.json right.json --output comparison.json`.
It returns a failing equality result for the observed OpenCV 4.10/4.12 drift at the unchanged tolerance.

No OAK hardware, Windows execution, XLink serialization, or end-to-end live sensor pipeline was tested.
The portable fallback and optional native build have distinct verification records. No hardware performance guarantee is made.

Future changes follow the [update workflow](update-workflow.md), also linked from `CONTRIBUTING.md`.

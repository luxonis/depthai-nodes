# Bundled MeasureAnything implementation

`MeasureAnythingDetector` and `MeasureAnythingDimensioner` share the frozen implementation in
[`depthai_nodes/algorithms/measure_anything/implementation/`](../../depthai_nodes/algorithms/measure_anything/implementation).
See the [public API overview](../../depthai_nodes/node/__init__.py),
[interactive measurement guide](../../depthai_nodes/algorithms/measure_anything/implementation/visualization.html)
and [replacement workflow](update-workflow.md).

## Identity and provenance

| Field                          | Selected value                               |
| ------------------------------ | -------------------------------------------- |
| Algorithm / configuration      | `fast_baseline_oak4_visibility_01/v002`      |
| Full candidate identity        | `algorithm-5e4abc3a09b54148950317cdeac60ba2` |
| Evaluated BoxLabRunners source | `d5e35e17d75f91368889dc53d950792eed4096e6`   |
| Committed evaluation evidence  | `d5964eee714737750787066fc24bc29744536d45`   |
| Visualization source           | `4b3853742c34a2d4548cfb599e05f37217dcec83`   |

The [manifest](../../depthai_nodes/algorithms/measure_anything/manifest.json) records
original source and candidate hashes, build artifact, dependencies, weights,
review, evaluation evidence and separate production-port hashes. The original
build is a native source archive, not an OCI image. Packaging does not establish
that the port is the evaluated artifact.

The original [evaluation receipts](../../depthai_nodes/algorithms/measure_anything/implementation/evaluation-evidence.json)
retain scoring policies, run/evaluation IDs, denominators, approval observations
and suppressed fields. The validator checks their linkage to the selected
candidate and run. Both nodes read `/configuration` from the original
[`scoped-settings.json`](../../depthai_nodes/algorithms/measure_anything/implementation/scoped-settings.json).
The manifest distinguishes its complete byte hash from the candidate's canonical
configuration hash. Settings remain private and frozen.

## Integration

MeasureAnything is the public API family name. The frozen candidate remains
`fast_baseline_oak4_visibility_01/v002`; renaming does not broaden its evaluated
object or input coverage. The packaged guide retains its original parcel-focused
examples and historical benchmark node labels. Source receipts and the numerical
comparison retain their original bytes and format identity.

The upstream factory becomes private `factory.py`; Runner imports use bundled
neutral scene/component types and the projector. Mechanical import relocation,
repository formatting and lazy SciPy loading preserve computation. Both nodes
use one binding and full identity, without public selectors or parameter overrides.
This candidate uses NumPy, SciPy and OpenCV, with no custom native kernels,
compiler, build flags or binaries. Existing box-node behavior is unchanged.

The packaged guide explains inputs, simulated processing, real validation
examples, accuracy and performance scopes. Four relative links become
commit-pinned upstream URLs. Documentation source/destination hashes and browser
receipts are recorded separately from computational provenance in the manifest.
Derived dimensional errors use this candidate's predictions and already-public
reference dimensions, cross-checked against official aggregates at absolute
tolerance 1e-10. Reference dimensions have 0.01 mm display precision. Held-out
median/P95 remain suppressed; reference poses and individual IoUs are unavailable.

## Verification coverage

The unchanged [source-versus-port comparison](comparison.json) covers 15 synthetic
scenes, three standalone fits and 18 authorized cached validation scenes. All
99 validation proposals and 19 eligible objects agree in membership, ordering,
decisions and selected-point order. Maximum geometry difference is zero at
absolute tolerance 1e-9 and relative tolerance zero. Cases include repeated scenes,
strided images, distortion, occlusion, no-fit and unknown height. These are decoded
array comparisons; the historical cache keeps its own manifest digest and is not
relabeled as the selected evaluated run. No fresh raw decoding, held-out input
comparison or Workbench evaluation is established by these checks.

Reference and port checks used macOS arm64, Python 3.12.2, DepthAI 3.12.0,
NumPy 2.2.6, SciPy 1.15.3 and OpenCV-headless 4.12.0.88. Numerical dependencies
match the frozen reference, which used Python 3.12.13. Recorded integration checks
include focused regressions, installed-wheel validation of 22 covered files,
both node bindings, 50 native SDK checks and a synthetic pipeline. Prior full-suite
failures were compared against an unchanged PR head; exact results remain in the
historical records linked below.

The latest recorded installed-guide check covers all 663 replay intervals,
application throughput arithmetic, 49 walkthrough stage/view combinations,
dimension and timing chart keyboard focus, pointer selection, both accuracy
splits and widths 390/768/1440 px. It passed offline without JavaScript errors
or external requests. Exact artifact hashes and browser observations identify
their original wheels; they do not verify a later rebuild.

## Limits and history

Guide timings distinguish three-parcel application throughput using predecoded
images and fresh inference from the original parcel-node intervals in a replay paced at 3 FPS.
They exclude camera/file decoding and display; node intervals also exclude upstream
inference and cloud generation. They establish neither camera-to-result latency,
maximum throughput nor a controlled scaling law with parcel count.

Other image grids retain a one-time warning without an equivalent accuracy claim.
Remote/XLink serialization is unsupported. The removed synthetic demo exceeded its
5 mm length tolerance (213.599998 mm versus 220 mm); removing it does not correct
that algorithm/fixture limitation. Algorithm corrections return to BoxLabRunners
for review and evaluation before another promotion.

[Historical integration and guide verification records](https://github.com/luxonis/depthai-nodes/blob/bc2f080f1efd9cb88e70207fd876e34918a84596/docs/parcel-promotions/implementation.md)
retain the earlier wheel hashes, environments, results and limitations.
[Predecessor promotion records and receipts](https://github.com/luxonis/depthai-nodes/tree/30e30d1eaac3561adcecda4f05110edae03982c0/docs/parcel-promotions)
retain their original candidate attribution. The predecessor's six native/visibility
findings are resolved in the
[upstream review-fix record](https://github.com/luxonis/BoxLabRunners/blob/d5964eee714737750787066fc24bc29744536d45/experiments/fast_baseline/v001/release-f14a597a366c46c6bdba00fa25e45bcd/2026-11-23-review-fixes/README.md).
Those native hazards do not describe the current NumPy implementation. Rollback
restores the complete code, settings, dependencies, assets and manifest together.

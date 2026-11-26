# MeasureAnything implementation and evidence

`MeasureAnythingDetector` and `MeasureAnythingDimensioner` share the frozen
Luxonis measurement algorithm in
[`implementation/`](../../depthai_nodes/algorithms/measure_anything/implementation).
Node class docstrings describe wiring, calibration, configuration and outputs:
[detector](../../depthai_nodes/node/measure_anything_detector/measure_anything_detector.py)
and [dimensioner](../../depthai_nodes/node/measure_anything_dimensioner/measure_anything_dimensioner.py).
The [interactive guide](../../depthai_nodes/algorithms/measure_anything/implementation/visualization.html)
explains the algorithm, real validation examples, accuracy and performance scopes.
Integration requirements are in [CONTRIBUTING.md](../../CONTRIBUTING.md#measureanything-integration).

## Frozen identity

| Field                             | Selected value                               |
| --------------------------------- | -------------------------------------------- |
| Algorithm / configuration         | `fast_baseline_oak4_visibility_01/v002`      |
| Candidate identity                | `algorithm-5e4abc3a09b54148950317cdeac60ba2` |
| Reference implementation revision | `d5e35e17d75f91368889dc53d950792eed4096e6`   |
| Reference evaluation revision     | `d5964eee714737750787066fc24bc29744536d45`   |

MeasureAnything is the public API family name. It does not broaden this
implementation's evaluated object or input coverage. Both nodes load
`/configuration` from the original
[`scoped-settings.json`](../../depthai_nodes/algorithms/measure_anything/implementation/scoped-settings.json).
Settings are frozen and have no public override API. Computation uses NumPy,
SciPy and OpenCV, without custom native kernels or a compiler.

The [manifest](../../depthai_nodes/algorithms/measure_anything/manifest.json)
separates reference candidate/source hashes from production-port file hashes.
The reference build is a source archive, not a container image. Import relocation,
neutral scene types, lazy dependency loading and public API naming are integration
adaptations; the port is not presented as the original evaluated artifact.

## Public evidence export

[`evaluation-evidence.json`](../../depthai_nodes/algorithms/measure_anything/implementation/evaluation-evidence.json)
is a derived public export. It retains scoring parameters, metric definitions,
run/evaluation identities, denominators, units, numerical results and suppressed
values. Nonpublic links and review administration metadata are removed. Schema
and metric-version labels use public names. Public policy digests are recomputed;
reference policy digests remain alongside them. The `public_export` block records
the original evidence checksum and these transformations.

The visualization is also a declared public export. Its embedded `public_export`
block records the original guide/evidence checksums. Predictions, point samples,
example geometry, reference dimensions, accuracy cross-checks and timing samples
are unchanged. Its links resolve to bundled settings, evidence, manifest and
public node source. No new evaluation or hardware measurement is added by export.

The validator checks both installed export hashes, reference-to-port mapping,
frozen settings, candidate/review/run/approval linkage and displayed metrics.
Original artifacts and the export process are retained by the algorithm maintainers;
users do not need access to those systems to use the package or inspect its evidence.

## Evaluation and integration coverage

The [retained numerical comparison](comparison.json) covers 15 synthetic scenes,
three standalone fits and 18 authorized decoded validation scenes. All 99
validation proposals and 19 eligible objects agree in membership, ordering,
decisions and selected-point order. Maximum geometry difference is zero at
absolute tolerance 1e-9 and relative tolerance zero. The cache has its own recorded
manifest identity; it is not relabeled as the evaluated dataset release.

Reference and port checks used macOS arm64, Python 3.12.2, DepthAI 3.12.0,
NumPy 2.2.6, SciPy 1.15.3 and OpenCV-headless 4.12.0.88. Numerical dependencies
match the reference, which used Python 3.12.13. Installed package integrity,
native SDK message checks, same-host pipeline integration and the guide's offline
interactions are verified separately from reference accuracy and throughput.

The public-export update reran the 15 synthetic scenes and three standalone fits
with identical outputs. The historical 18-scene comparison was not rerun. The
installed wheel passed 50 native SDK checks and offline guide checks covering
keyboard focus, portable links, displayed metrics and 663 retained replay samples.

Full evaluation inputs are not distributed. Synthetic integration fixtures do not
reproduce the complete benchmark. Previously published reference dimensions have
0.01 mm display precision. Held-out median/P95 remain suppressed; individual
reference poses and IoUs are unavailable. Derived error plots do not add new
reference labels or evaluation results.

## Limits

Guide timings distinguish three-parcel application throughput using predecoded
images and fresh inference from parcel-node intervals in replay paced at 3 FPS.
They exclude camera/file decoding and display; node intervals also exclude upstream
inference and cloud generation. They establish neither camera-to-result latency,
maximum throughput nor a controlled scaling law with object count.

Other image grids retain a one-time warning without an equivalent accuracy claim.
Nested Python messages have no XLink or RemoteConnection serialization. The
removed synthetic demo exceeded its 5 mm length tolerance (213.599998 mm versus
220 mm); removal does not correct that algorithm/fixture limitation. Changes to
computation or settings need review and evaluation before another integration.
Rollback restores code, settings, dependencies, assets and manifest together.

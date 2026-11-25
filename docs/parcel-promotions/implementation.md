# Bundled parcel implementation

`ParcelDetector` and `ParcelDimensioner` use one frozen implementation in
[`depthai_nodes/algorithms/parcel/implementation/`](../../depthai_nodes/algorithms/parcel/implementation).
For node usage, see the [public API overview](../../depthai_nodes/node/__init__.py)
and [measurement guide](../../depthai_nodes/algorithms/parcel/implementation/visualization.html).
For replacements, follow the [update workflow](update-workflow.md).

## Selected candidate and provenance

| Field                          | Selected value                               |
| ------------------------------ | -------------------------------------------- |
| Algorithm / configuration      | `fast_baseline_oak4_visibility_01/v002`      |
| Full candidate identity        | `algorithm-5e4abc3a09b54148950317cdeac60ba2` |
| Evaluated BoxLabRunners source | `d5e35e17d75f91368889dc53d950792eed4096e6`   |
| Committed evaluation evidence  | `d5964eee714737750787066fc24bc29744536d45`   |
| Visualization source           | `4b3853742c34a2d4548cfb599e05f37217dcec83`   |

The [package manifest](../../depthai_nodes/algorithms/parcel/manifest.json) records
original source hashes, build artifact, dependencies, weights, candidate hashes,
review, validation and held-out evidence, and separate production-port hashes.
The original build is a native source archive, not an OCI image.
[Evaluation receipts](../../depthai_nodes/algorithms/parcel/implementation/evaluation-evidence.json)
retain the complete policy, run/evaluation IDs, denominators and suppressed fields.
Their historical approval observations are preserved exactly and checked against
the selected candidate and run. These records are the authoritative evidence;
relocating or packaging the code does not make the port the evaluated artifact.

Both nodes load the nested `configuration` object from the original
[`scoped-settings.json`](../../depthai_nodes/algorithms/parcel/implementation/scoped-settings.json),
without a separately serialized parameter file. The manifest hashes the complete
settings bytes and records `/configuration` separately from the candidate's
canonical configuration hash. Settings remain private and frozen.

## Integration adaptations

The source factory becomes private `factory.py`; Runner imports use bundled
neutral scene/component types and the projector. Repository formatting and lazy
SciPy loading preserve computation. Both nodes use the same internal binding and
full identity. There is no public algorithm selector or parameter override API.

This candidate uses NumPy, SciPy and OpenCV, with no custom native kernels,
compiler, build flags or binaries. Predecessor C++ metadata is rejected by the
validator. Public APIs, synchronization, calibration conversion, optional support
plane, immutable parcel point storage and existing box-node behavior are retained.

The packaged measurement guide adapts four links to commit-pinned upstream URLs.
Source and destination HTML hashes are recorded separately from computational
provenance. Its real validation examples derive this candidate's errors from
already-published reference dimensions, cross-checked against official aggregates
at absolute tolerance 1e-10. References have 0.01 mm display precision. Held-out
median/P95 remain suppressed; reference poses and individual IoUs are unavailable.
Keyboard activation restores focus after chart redraw; pointer selection is retained.
These documentation changes add no algorithm evaluation or hardware measurement.

## Verification scope

The byte-preserved [numerical comparison](comparison.json) covers 15 synthetic
scenes, three standalone fits and 18 already-authorized cached validation scenes
against the frozen source. All 99 validation proposals and 19 eligible objects
agree in membership, ordering, decisions and extracted point order. Maximum
geometry difference is zero with absolute tolerance 1e-9 and relative tolerance
zero. Repeated scenes, strided images, distortion, occlusion, no-fit and unknown
height are covered. This is decoded-array parity. The historical cache has its
own recorded manifest digest; it is not relabeled as the selected run manifest.
No fresh raw decoding, held-out input comparison or Workbench evaluation is claimed.

Port checks used macOS arm64, Python 3.12.2, DepthAI 3.12.0, NumPy 2.2.6,
SciPy 1.15.3 and OpenCV-headless 4.12.0.88. Numerical dependencies match the frozen
reference; its Python patch version was 3.12.13. Focused parcel/reference/box
checks, installed-wheel validation, 50 native SDK message checks and a synthetic
pipeline through both public nodes passed. Full-suite failures were reproduced
on the unchanged PR head; exact historical results remain in the archive below.

The customer-guide wheel, SHA-256
`b5205b498605be7e38661778738d9e242a82bbf3337335fc44952e1f8430f12e`,
validated 22 covered files outside both source trees. Its installed page passed
49 stage/view combinations, 24 keyboard and 12 pointer cases, separate split/error
selectors, aggregate arithmetic, minimum/maximum validation ranking and widths
390/768/1440 px offline, with no external requests or JavaScript errors.
These are observations of that exact historical wheel, not a later rebuild.

The documentation consolidation rechecked 87 focused tests and 46 subtests
(one optional Open3D skip), API documentation and a fresh installed wheel:
22 package hashes, both node bindings and 50 native SDK checks pass. All covered
package files and the numerical comparison remain byte-identical to the preceding
revision; only manifest documentation references change within the package.

## Limits and historical evidence

The recorded upstream 11.18 FPS has a predecoded-image/fresh-inference scope;
it does not measure transport, raw decoding or live throughput of this package.
Other image grids retain the documented one-time warning without an equivalent
accuracy claim. Remote/XLink serialization is unsupported. No new Windows,
Linux or OAK run is established by this documentation consolidation.

The synthetic example's existing 5 mm assertion fails before and after the
settings-file simplification: length 213.599998 mm versus expected 220 mm
(width 158.419998 mm, height 110 mm). Packaging preserves this result; algorithm
corrections require upstream review and evaluation.

[Historical promotion records and complete verification receipts](https://github.com/luxonis/depthai-nodes/tree/30e30d1eaac3561adcecda4f05110edae03982c0/docs/parcel-promotions)
retain their exact candidate identities and artifact hashes, including the
predecessor `fast_baseline/v001` and its native builds. Released artifacts retain
their original bytes. Earlier scores, timings and browser observations do not
verify a replacement. The predecessor's six native/visibility findings, including
both allocation hazards, are resolved in the
[upstream review-fix record](https://github.com/luxonis/BoxLabRunners/blob/d5964eee714737750787066fc24bc29744536d45/experiments/fast_baseline/v001/release-f14a597a366c46c6bdba00fa25e45bcd/2026-11-23-review-fixes/README.md).
The [OAK-4 follow-up](https://github.com/luxonis/BoxLabRunners/blob/d5964eee714737750787066fc24bc29744536d45/experiments/oak4_visibility_10fps/FOLLOWUP.md)
records the subsequent algorithm work. These findings concern earlier candidates;
the current implementation contains no custom C++ kernels.

## Support-plane and timing guide update

The current source guide is pinned above. Its simulated parcel rests at z=0,
with support samples on that plane surrounding all four sides of its footprint.
The unrelated orange point cluster is removed; occlusion hides far-side samples
behind the parcel. The walkthrough remains explicitly simulated.

The speed section contains this candidate's timings only. Its 54 measured
validation runs (18 scenes × three repetitions) average 35.25630757407408 ms/scene;
fitting time divided by 57 fitting calls is 4.860072929824561 ms/parcel. An
interactive plot separates whole-scene time, amortized scene cost per measured
parcel and fitting-only cost. The count table has 17 one-parcel scenes and one
two-parcel scene. Geometry, input points and rejected proposals differ, so no
controlled count scaling curve or extrapolation is claimed. The separate
three-parcel wrapper workload averages 51.194926545 ms/batch over 200 batches
after ten warm-ups, or 17.064975515 ms/parcel amortized. Its different timer
boundaries and full-application throughput are explained separately.

Source generation and browser receipts are linked from
`port_verification.surface_and_timing_visualization`. Earlier receipts retain
their original observations through the historical links above. Computation,
settings, evaluation receipts, official scores and candidate identity are unchanged.

Verification of this update: all 41 parcel algorithm tests and pre-commit pass.
The freshly built wheel, SHA-256
`faf53e669e503eda225dfa6f7ae98fb9f58c3867c901ec8dedd0f0238fc6e5a7`,
validates all 22 covered files outside both source trees, preserves the original
settings file and `/configuration` pointer, and contains exactly the source
evidence JSON. Comparing package bytes with PR head
`da95b7be7c88a4473d2824dc9ba1c28bb19d5453` changes only the HTML guide and
manifest. Computation, frozen settings, receipts and validator are byte-identical.
The installed guide passes 49 stage/view combinations, 24 dimension-chart keyboard
cases, six timing-chart keyboard cases with retained focus, 12 pointer cases,
all three timing views, support-plane geometry checks, arithmetic/ranking and
responsive widths 390/768/1440 px. Chrome 158.0.8127.88 runs offline with zero
JavaScript errors or external requests. The installed HTML SHA-256 is
`2c5d6e33511cac821fafc826f7e2082f97341649c7d02b95d9b793bb92659fbe`.

```sh
python -m pytest tests/unittests/test_parcel_algorithm -q
pre-commit run --all-files
python -m build --wheel --no-isolation
# Install the wheel with [parcel] into a fresh environment outside both repositories.
# Run BoxLabRunners' visualisation/check.cjs against the installed HTML offline.
```

## Replay performance guide update

The guide now leads with the retained three-parcel application result: 671
completed frames in a fixed 60-second window after ten seconds of warm-up,
11.183333333333334 frames/s. The builder independently reproduces the count,
all-three-parcels measured status, unique sequences and timestamps from the
original prepared-application timeline. Four predecoded recorded image inputs
feed fresh NeuralDepth LARGE, point cloud generation, controller and parcel nodes.
Camera/file decoding and browser rendering are outside the benchmark. Throughput
is not per-frame latency.

A separate chart recovers detector-entry-to-measurement-callback-arrival wall
times from all 663 native replay outputs. Detector start is its logged monotonic
end time minus its recorded duration. Every ordered single-inflight pair must
match sequence IDs, have a successful result after its own start and before the
next detector start. Repeated sequence IDs are preserved. Measurement arrival
can precede the detector's finally log; joining solely on log completion order
would discard one valid output. Source instrumentation, candidate identities and
shared session code hashes are checked.

| Reported parcels | Frames | Mean node interval ms | Median ms |      P95 ms |
| ---------------- | -----: | --------------------: | --------: | ----------: |
| 1                |    343 |    48.125655379008755 | 46.380065 |  58.2414152 |
| 2                |    226 |     55.22905627876106 | 55.027959 |  65.1538215 |
| 3                |     80 |     62.63724366250001 | 62.214069 | 69.54066335 |

All outputs, including startup and 14 with zero/four/five parcels, are retained in
embedded evidence. No warm-up or outlier filtering is applied. Frames come from
one recording and are correlated. Counts are algorithm-reported, not independent
field labels. The native replay was paced at 3 FPS. This interval includes parcel
node conversion/checks, detection, visibility, extraction, fitting and scheduling;
it excludes upstream inference, point-cloud generation, controller input
preparation and subsequent display. It does not establish full-pipeline latency
or maximum throughput. One/two-parcel full-pipeline FPS and camera-to-result
latency remain unavailable. Scene geometry/visibility also vary, so no controlled
count scaling law is claimed.

Mean/median/P95 replay controls are separate from the validation timing views.
The source generation/browser receipts and exact source commit are linked from
`port_verification.replay_processing_visualization`. Frozen computation, settings,
evaluation receipts, scores and candidate identity are unchanged; no new hardware
run or Workbench evaluation is added. Earlier guide checks remain historical.

Verification of the replay guide: all 41 parcel algorithm tests and pre-commit
pass. A fresh installed wheel, SHA-256
`1a472655b44259bde69e80b0b449abf037ef2027a59035898244f7c05a007687`,
validates all 22 covered files outside both source trees. Source and installed
evidence JSON are identical. Package bytes compared with PR head
`ee676729571544206aef6593afec999f709bdbc3` differ only in the HTML guide
and manifest; computation, original settings, receipts and validator are unchanged.
The installed HTML SHA-256 is
`c195176e3d77e4df433ea824bb219c7a91ef17412792635baf106ef6c6a2ae32`.
Its offline Chrome 158.0.8127.88 check verifies every one of the 663 paired
intervals, mean/median/P95 by count, 671/60 application throughput, 49 walkthrough
stage/view combinations, 24 dimension-chart keyboard cases, six validation-timing
and six replay-chart keyboard cases, 12 pointer cases, both accuracy splits and
responsive widths 390/768/1440 px. No JavaScript errors or external requests occur.

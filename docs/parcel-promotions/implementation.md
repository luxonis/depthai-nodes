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
| Visualization source           | `515ded0ebc24ea91f74d1555a346314c41cf5310`   |

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

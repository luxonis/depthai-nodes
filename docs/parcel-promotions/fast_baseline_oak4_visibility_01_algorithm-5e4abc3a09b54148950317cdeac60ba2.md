# OAK-4 visibility v002 promotion

Operator selection on November 25, 2026 replaces `fast_baseline/v001`, candidate
`algorithm-2f65a6f6296d48e981f5fe3f77bfde89`, in PR #348 with
`fast_baseline_oak4_visibility_01/v002`, candidate
`algorithm-5e4abc3a09b54148950317cdeac60ba2`. Both public parcel nodes retain their
APIs and the same scene conversion, synchronization, plane handling, and immutable
batch ownership. Existing box nodes are unchanged. This is a prepared PR update;
no package is published and the PR is not merged.

## Frozen upstream

- Evaluated source: BoxLabRunners `d5e35e17d75f91368889dc53d950792eed4096e6`.
- Evidence and visualization: `d5964eee714737750787066fc24bc29744536d45`, merged PR #19.
- [Completed record](https://github.com/luxonis/BoxLabRunners/blob/d5964eee714737750787066fc24bc29744536d45/experiments/fast_baseline_oak4_visibility_01/v002/release-f14a597a366c46c6bdba00fa25e45bcd/2026-11-25-held-out/record.json).
- [Review and held-out follow-up](https://github.com/luxonis/BoxLabRunners/blob/d5964eee714737750787066fc24bc29744536d45/experiments/oak4_visibility_10fps/FOLLOWUP.md).
- Build SHA-256 `bad3b09ff518e15b05db814bdc8bb40d26818328f38a1dddf78d4fc9523d322c`
  identifies a native source archive, not an OCI image. The manifest leaves image
  digest null and verifies archive kind and digest separately.
- Complete settings are byte-identical to upstream
  `algorithms/fast_baseline_oak4_visibility_01/configurations/v002.json`.
  `configuration.json` serializes the unchanged parameter object. Three projection
  workers, sequential candidate validation, stride-one boundary probes, sampling,
  geometry, and all thresholds remain frozen.

Validation sealed with 18/18 snapshots, 19/19 matches, no false positives, misses,
or failed snapshots, and 8/19 within 5 mm on all axes. Approved held-out test sealed
with 17/17 snapshots and matches, no false positives, misses, or failures, and 8/17
within tolerance. Test length/width/height MAE is
4.414301767905942 / 3.809994109376463 / 2.3389752628108735 mm.
The complete policy, metric version, run/evaluation IDs, denominators, review,
suppressed fields, and original source hashes are in the manifest and packaged
`evaluation-evidence.json`. Individual reference findings remain unavailable;
exact test medians and P95 errors stay suppressed. No new evaluation or tuning
followed held-out exposure.

## Source to port

`__init__.py` becomes private `factory.py`; `geometry.py`, `visibility.py`, and
`numpy_ops.py` retain their computation. Only Runner imports become bundled
neutral types/projector imports, followed by repository formatting. The existing
lazy entry point keeps SciPy optional until components are constructed. The
complete source file map includes the original Runner and dependency lock; the
package excludes transport, journals, registration, and credentials. Neutral
`scene.py` and `components.py` remain at their stable package paths.

The selected algorithm has no custom native kernels. Obsolete `native.py`,
`kernels.cpp`, and `build_native.py` are removed. This wheel needs no compiler.
Both nodes bind to the new full candidate through `parcel.ALGORITHM_ID` and the
same private configuration. No public selector or settings API is added.

The visualization is copied to
`depthai_nodes/algorithms/parcel/implementation/visualization.html` with four
relative links resolved to commit-pinned upstream URLs. Its embedded evidence,
predictions, case walkthroughs, and interactions remain unchanged. Source and
installed destination HTML hashes are separate manifest fields. Quality-ranked
best/worst examples and individual error plots remain explicitly unavailable;
prediction plots do not imply individual accuracy evidence.

## Port verification

[Machine-readable comparison](oak4-visibility-comparison.json) compares 15
synthetic scenes, three standalone measurement cases, and all 18 already-authorized
cached validation scenes against the exact frozen source. Membership, candidate
ordering, decisions, and extracted point order agree exactly, including all 99
validation proposals and 19 eligible objects. Maximum geometry difference is zero
at declared absolute tolerance 1e-9 and relative tolerance zero. Repeated scenes,
strided images, distortion, clipped/occluded parcels, no-fit, and unknown height
are covered. This is decoded-array parity, not a fresh raw-decoding or Workbench
accuracy evaluation. The historical cache manifest digest is retained in the
report; it is not relabeled as the evaluated run manifest.

Reference and target use Python 3.12.2 on macOS arm64, NumPy 2.2.6, SciPy 1.15.3,
OpenCV 4.12.0 (distribution 4.12.0.88). The frozen native evaluation recorded
Python 3.12.13; this port check uses the exact numerical dependency pins but a
different Python patch version. Native message tests use the PR's exact SDK
3.12.0. Existing image-profile warning behavior remains; other resolutions or
rectification policies have no new accuracy claim.

Commands (reference root contains exact evaluated tracked files plus packaged
`evaluation-evidence.json` copied as `fast-baseline.json` and `scoped-settings.json`
copied as `fast-baseline-settings.json`, the tool's legacy filenames):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/verify_parcel_port.py \
  --reference-root "$REFERENCE_ROOT" --validation-cache "$VALIDATION_CACHE" \
  --output docs/parcel-promotions/oak4-visibility-comparison.json
python -m pytest tests/unittests/test_parcel_algorithm tests/unittests/test_parcel_nodes tests/parcel_reference -q
python -m pytest tests/unittests
pre-commit run --all-files
pydoctor depthai_nodes
python -m build --wheel --no-isolation
```

Verification completed. The [installed-wheel receipt](oak4-visibility-wheel.json) records the final artifact digest and HTML hashes; the [offline browser report](oak4-visibility-visualization.json) records interaction coverage.

- 70 focused parcel/reference/box tests and 46 subtests pass; one existing optional
  Open3D-dependent box test is skipped.
- The clean universal wheel contains all 23 manifest-covered files, the selected
  visualization and frozen settings, and no obsolete candidate directory, custom
  kernel sources, or native binary. Its installed manifest validates outside both
  repositories with no source checkout on the import path.
- All 50 native DepthAI message/node checks pass from that installed wheel.
  A separate actual synthetic pipeline runs the bundled finder and measurement
  through both public nodes, produces one parcel, and matches direct computation
  at absolute tolerance 1e-9: 176.46833038330078 × 154.15728759765625 × 110.0 mm.
  This small synthetic image grid emits the existing expected profile warning;
  it is conversion/pipeline coverage, not accuracy evidence for another grid.
- The installed visualization passes offline Chrome checks at widths 390, 768,
  and 1440 px, all 49 stage/view combinations, crop/keyboard controls, separate
  validation/test prediction and aggregate selectors, suppressed values, and
  portable links. No external requests or JavaScript errors occur. Both embedded
  evidence datasets agree with the original source page.
- All four ported computation ASTs match the frozen source after mechanical
  import relocation, docstring removal, and repository formatting. Exact-type
  checks retain their original semantics with scoped lint suppressions.
- Pre-commit and API documentation with warnings as errors pass.

The final full-unit run has 580 passes, 50 failures, 93 errors, and six skips.
The unchanged PR head has 576 passes, 51 failures, 93 errors, and six skips.
Every final failing/erroring test ID also fails on the unchanged head; there are
no new failing test IDs. An earlier replacement run reproduced all 144 baseline
failure/error IDs. The final run does not reproduce the unchanged axial-depth
`test_deterministic_host_pipeline` failure; no axial-depth code was changed.
These collection/mock/model-cache failures remain limitations of the full suite.

Historical promotion records remain unchanged; their results are not reused as
verification of this replacement.

## Limits and rollback

No new OAK hardware performance run, Windows run, raw artifact decoding, training
labels, or held-out inputs were accessed. The recorded 11.18 FPS is the upstream
experiment's predecoded-image/fresh-inference processing scope; it is not a claim
for this newly built wheel. Remote/XLink serialization remains unsupported.
Rollback restores the complete previous source/settings/assets/manifest revision,
not only its candidate ID. See [future update workflow](update-workflow.md).

## Provenance review corrections

The [review-fix record](oak4-visibility-provenance-review.md) corrects the inherited held-out approval and native-build metadata after the initial promotion. Original accuracy, comparison, and wheel receipts above describe that earlier artifact and are retained unchanged. The frozen computation, settings, and evaluation receipts are unchanged.

## Visualization keyboard focus

The [keyboard-focus record](oak4-visibility-keyboard-focus.md) documents the regenerated upstream page, keyboard-only focus restoration, preserved pointer selection, and current source/destination HTML hashes. Earlier browser and wheel receipts describe their original artifacts. Frozen computation, settings, and embedded evidence remain unchanged.

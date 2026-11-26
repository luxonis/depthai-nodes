# Update the bundled MeasureAnything algorithm

BoxLabRunners owns algorithm development, review, tuning and evaluation.
DepthAI Nodes owns public APIs, message conversion, packaging and integration.
Return computation or frozen-settings findings to BoxLabRunners with a reproduction
and their effect. If ownership is unclear, record the uncertainty before changing
the frozen dependency closure. Packaging and documentation corrections can be
verified here without claiming a new algorithm evaluation.

The [current implementation record](implementation.md) identifies the selected
candidate and actual verification coverage. Read the commit-pinned BoxLabRunners
[design](https://github.com/luxonis/BoxLabRunners/blob/2cb32ae03881d4b897d15512c9850b1790fae703/docs/depthai-nodes/design.md)
and [promotion runbook](https://github.com/luxonis/BoxLabRunners/blob/2cb32ae03881d4b897d15512c9850b1790fae703/docs/depthai-nodes/promote.md)
before selecting a replacement. Node contracts and input requirements live in the
[public API overview](../../depthai_nodes/node/__init__.py) and node docstrings.
Existing box nodes have a separate contract.

1. Select a newly reviewed and evaluated BoxLabRunners candidate. Pin the evaluated
   source commit and committed evidence revision, rather than the checkout's `HEAD`.
   Verify source hashes, original settings bytes and independent review resolutions.
   Record algorithm name, configuration version and full candidate identity.
   Computational or settings changes require a new candidate; additional evaluation
   of unchanged code/settings adds evidence to the existing version.
1. Replace the contents of the stable
   `depthai_nodes/algorithms/measure_anything/implementation/` directory. Include the complete
   computational dependency closure, original `scoped-settings.json`, required
   assets, attribution and any native source/build instructions. Load its nested
   `configuration` directly. Exclude Workbench transport and credentials.
   Update code, settings, dependencies, binding and manifest together. Keep one
   implementation directory; Git history and retained release artifacts preserve
   previous implementations.
1. Bind both nodes to the same frozen implementation, configuration and full
   identity. Preserve the public API, point order, numerical behavior, runtime
   support plane/config handling, calibration conversion and native/fallback
   support where applicable. Keep settings private, without selectors or overrides.
1. Compare source and port on identical synthetic and already-authorized validation
   inputs. Declare tolerances first. Check proposal order, indices, eligibility,
   selected-point order, geometry, no-fit and unknown-height outcomes. Exercise
   repeated scenes, exception cleanup and native/fallback paths when present.
   State unavailable assets and comparison limits. Fixtures do not establish full
   dataset parity; fetching held-out inputs or reading training labels needs
   separate authorization.
1. Run focused algorithm, message, node, reference and relevant box tests. Use real
   SDK messages to check delivered-pixel calibration, units/transforms, synchronized
   timing, config application, unset-plane behavior, failed/empty frames, immutable
   queued ownership and identity mismatch. The dimensioner uses the plane carried
   by the batch. Preserve existing input-profile warnings and frozen settings.
   Run the full unit suite, pre-commit and API documentation build; compare failures
   against the unchanged base and report existing failures explicitly.
1. Update `depthai_nodes/algorithms/measure_anything/manifest.json`. Keep evaluated-source
   provenance separate from production-port hashes and adaptations. Preserve source
   commit, candidate hashes, original image or build artifact kind/digest, dependency
   lock, weights and frozen settings. Preserve complete review, scoring policy,
   metric version, evaluation/run IDs, denominators, null reasons and suppression.
   Record the settings-file hash and `/configuration` pointer separately from the
   candidate's canonical configuration hash. Hash raw installed package bytes with
   repository-relative `depthai_nodes/` keys. Exclude the manifest and optional
   local binaries from that map; record native builds separately.
1. Copy the selected visualization/assets. Repair links to shipped files or
   commit-pinned sources. Record its documentation revision and source/destination
   hashes independently of computation. Verify the installed page offline against
   the selected candidate/settings/evidence, including keyboard focus, responsive
   layout and interactions. Documentation updates do not imply reevaluation.
1. Build a wheel from a clean staging tree, then install it outside both source
   repositories. A reused `build/lib` can retain removed packages. Check wheel
   contents for obsolete directories/assets/binaries, validate all declared hashes,
   verify lazy optional imports and both node bindings, and run native SDK checks
   plus a synthetic pipeline through both nodes. Record platform/dependency
   differences. If the candidate uses custom native libraries, build from its
   selected source, test a clean fallback, and exclude local binaries from a
   universal wheel.
1. Replace `implementation.md` with the new candidate's concise integration record
   and `comparison.json` with its actual comparison evidence. Keep these three
   documentation files. Put detailed development receipts in retained review/build
   artifacts, with exact hashes and immutable links from the implementation record.
   Before replacement, pin the earlier record/evidence and retain exact released
   wheels, manifests, settings and assets. Never transfer old scores or performance
   claims to a newly built artifact. Update README and manifest references together.

Useful commands (install `.[dev,box-measurement,measure-anything]` first):

```sh
python -m pytest tests/unittests/test_measure_anything_algorithm tests/unittests/test_measure_anything_nodes tests/measure_anything_reference -q
python -m pytest tests/unittests
pre-commit run --all-files
pydoctor depthai_nodes
python -m build --wheel --no-isolation
```

`tests/measure_anything_reference/verify_measure_anything_port.py` performs the repeatable
frozen-source comparison;
`tests/measure_anything_reference/` supplies repository fixtures. Its reference root must
contain the exact evaluated files and evidence. Record the command, asset hashes,
environment and coverage in `comparison.json`; keep temporary reports outside the
repository. Preserve prior published evidence unchanged in Git/release artifacts.

Rollback restores the complete previously verified release: code, settings,
dependencies, assets and manifest. Restoring only an identity or binding would
misattribute the installed computation. Publishing or merging requires separate
authorization.

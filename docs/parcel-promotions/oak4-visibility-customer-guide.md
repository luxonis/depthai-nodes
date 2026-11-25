# Customer-facing parcel measurement guide

The packaged `implementation/visualization.html` is updated from BoxLabRunners
documentation commit `515ded0ebc24ea91f74d1555a346314c41cf5310`. The algorithm,
frozen configuration, candidate identity, dependency pins and official Workbench
receipts are unchanged. No algorithm evaluation or hardware measurement is added.
The manifest records separate source and destination HTML hashes; packaging
adapts only four relative links to commit-pinned upstream URLs.

The guide explains inputs, the simulated workflow, two real accuracy examples,
overall accuracy, scoped OAK-4 performance and practical limitations. Customer
narrative and photographs lead; configuration and provenance are expandable.

## Real accuracy examples and derived evidence

The previous source page could not rank accuracy cases from aggregate receipts
alone. The new builder uses reference dimensions already publicly disclosed in
BoxLabRunners' historical fast-baseline visualization at commit
`96a2c2d781845eda3f7f64ae0f151263084aa499`, for the same frozen release.
It reuses only snapshot/prediction identifiers and reference dimensions; no
historical candidate prediction, error, pass flag or overlap score is reused.
This candidate's own 19 validation and 17 test predictions provide the measured
values. Full-split MAE, signed bias, mean relative absolute error, object counts,
and all-axis success counts must reproduce its official receipts at absolute
cross-check tolerance 1e-10, including the two-parcel snapshot.

These are independently checked derived comparisons, not a new private findings
export. Published reference dimensions have 0.01 mm display precision. Official
held-out median/P95 fields remain suppressed, and individual reference poses and
per-object IoUs remain unavailable. The source page records the reference source
commit, SHA-256, disclosure provenance and derivation method.

Among all 18 validation snapshots (19 matched parcels), rank a snapshot by its
largest absolute L/W/H error, taking its worst parcel in multi-object scenes.
All references matched, with no failed snapshots. The two real examples are:

- V03, `snapshot-5c6e5c7d4600416a9f463ff6dc5f820a`: largest error
  1.8992038561340792 mm, all dimensions within ±5 mm.
- V15, `snapshot-8e1d51f1afbc4122816c6ff00a5a5302`: width overestimate
  32.26668251461706 mm; length and height remain within tolerance.

Each viewer uses actual hash-verified validation images and point samples.
Instrumented proposal membership/decisions and cuboids must equal uninstrumented
frozen execution; traced dimensions must match saved predictions at 1e-9 absolute
tolerance. The difficult case identifies a footprint-width error without inventing
its physical cause or a reference-pose overlay. No new held-out input access,
training labels or private operator storage is needed.

## Verification and packaging

The source browser regression checks arithmetic against the official aggregates,
true minimum/maximum validation ranking, all 49 schematic/real stage-view
combinations, 24 keyboard and 12 pointer cases, separate split/error selectors,
suppressed P95, links, and responsive widths 390/768/1440 px. It passes offline
with no JavaScript errors or external requests. Both source and installed pages
use the same regression driver. All 41 parcel algorithm tests pass after integrating the latest settings-file packaging change.

The [installed browser receipt](oak4-visibility-customer-browser.json) and
[clean-wheel receipt](oak4-visibility-customer-wheel.json) record the installed
artifact's checks and hashes. Earlier guide, browser and wheel receipts retain
their original observations. The original promotion record's unavailable-case
statements describe that earlier page; the public reference derivation above is
the new evidence that makes these accuracy examples possible.

```sh
python -m pytest tests/unittests/test_parcel_algorithm -q
pre-commit run --all-files
python -m build --wheel --no-isolation
NODE_PATH=/path/to/node_modules node /path/to/BoxLabRunners/experiments/oak4_visibility_10fps/visualisation/check.cjs \
  /tmp /path/to/installed/depthai_nodes/algorithms/parcel/implementation/visualization.html \
  docs/parcel-promotions/oak4-visibility-customer-browser.json
```

The clean installed wheel passes the complete customer-guide browser regression, including both split arithmetic checks and minimum/maximum validation ranking. Its 22 manifest-covered files validate outside both source trees. A byte comparison with the preceding PR commit confirms that only the HTML visualization and manifest changed; computation, settings, receipts and validator are byte-identical. Pre-commit passes.

The PR advanced to `46635b16701077d58510baaa09c53b3d5901692d` during this update. Its single-settings-file refactor is preserved: the runtime reads the original `scoped-settings.json` configuration object directly, and no separate `configuration.json` is reintroduced. Installed checks and package-byte comparisons are repeated on the integrated revision.

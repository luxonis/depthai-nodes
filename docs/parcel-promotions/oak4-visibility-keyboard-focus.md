# Visibility chart keyboard focus

Fixes [CodeRabbit finding 4693105588](https://github.com/luxonis/depthai-nodes/pull/348#discussion_r4693105588).
Enter or Space replaced the activated prediction marker while redrawing the charts,
so keyboard focus fell back to the document body. The handler now remembers the
chart axis and record before redraw, then focuses the matching replacement marker
only for keyboard activation. Pointer activation still updates selection without
restoring focus.

The fix originates in BoxLabRunners `experiments/oak4_visibility_10fps/visualisation/page.js`.
The canonical page was regenerated and copied from documentation commit
`96a2c2d781845eda3f7f64ae0f151263084aa499`. Only four relative links are adapted
to commit-pinned upstream URLs when packaging. Both embedded JSON datasets remain
byte-identical to the previous page, including cases, predictions, scores, and
candidate identity. Computation, frozen settings, receipts, and algorithm hashes
are unchanged; this is a documentation correction, with no new Workbench run.

The manifest records the separate source and destination HTML hashes and pins
the new source documentation commit independently of the frozen computation commit.
Earlier browser and wheel receipts remain historical observations.

## Verification

The new regression fails on the original page at validation/predictions/axis 0/Enter.
The regenerated source page passes 24 keyboard combinations (Enter/Space, three
axes, validation/test splits, prediction/unavailable-error modes) and 12 pointer
cases. Synthetic pointer dispatch must leave existing focus alone; real clicks
must still select another prediction. The existing 49 stage/view combinations,
aggregate/suppression controls, portable links, and widths 390/768/1440 px also
pass offline with zero JavaScript errors and external requests.

Use the same committed BoxLabRunners browser check against the installed page:

```sh
NODE_PATH=/path/to/node_modules node /path/to/BoxLabRunners/experiments/oak4_visibility_10fps/visualisation/check.cjs \
  /tmp /path/to/installed/depthai_nodes/algorithms/parcel/implementation/visualization.html \
  docs/parcel-promotions/oak4-visibility-keyboard-browser.json
python -m pytest tests/unittests/test_parcel_algorithm -q
pre-commit run --all-files
python -m build --wheel --no-isolation
```

The [installed browser receipt](oak4-visibility-keyboard-browser.json) and
[wheel receipt](oak4-visibility-keyboard-wheel.json) record checks for this update.
The installed page needs neither source checkout nor network for its content or
interactions; the browser test driver is external test tooling.

The clean installed wheel passes the same 24 keyboard and 12 pointer cases, all 49 stage/view combinations, and the responsive/offline checks. All 39 parcel algorithm tests and pre-commit checks pass. Wheel comparison confirms that only the visualization and manifest changed; computation, settings, receipts, and validator bytes are identical to the provenance-fix wheel.

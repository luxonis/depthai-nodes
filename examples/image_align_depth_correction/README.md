# Verify RGB axial-depth correction

This example runs deterministic replay through ImageAlign, the shared
correction node, and PointCloud. It uses different source and target grids and
a rotated, translated source camera. No live camera or model is started.

The program asserts that ImageAlign retains source Z, that correction changes
the target Z across the image, and that the resulting RGB-frame cloud recovers
the source plane within quantization tolerance. A contradiction fails the run.
It prints a JSON verification record with the policy, method, conventions,
package location, SDK version, geometry, placement, and measured residuals.
It checks behavior on the installed SDK rather than requiring an exact SDK
version. The OAK-4 app requirements retain the versions used for its recorded
hardware verification.

## Run the host fixture

Build from the repository root and install the wheel into an isolated
environment. Run the copied example outside the checkout.

```bash
python -m build --wheel --outdir /tmp/axial-dist
python -m venv /tmp/axial-installed
/tmp/axial-installed/bin/python -m pip install /tmp/axial-dist/depthai_nodes-*.whl
cp examples/image_align_depth_correction/main.py /tmp/axial-example.py
(
    cd /tmp
    oakctl run-script /tmp/axial-installed/bin/python -I /tmp/axial-example.py --report /tmp/axial-host.json
)
```

Expected output includes `source_z_retained_by_image_align=true` and a maximum
corrected source-plane residual below 0.6 mm. The uncorrected residual exceeds
50 mm. ImageAlign and PointCloud run on the workstation CPU.

For artifact verification, copy
`tests/unittests/test_axial_depth/verify_axial_depth.py` outside the checkout and
run it with the installed environment. Supply `--wheel`,
`--example`, `--revision`, and `--report`. It compares installed package files
with the wheel, runs this fixture in isolated Python mode, and adds the wheel
SHA-256 and source revision to the report. Editable installs fail this check.

## Run inside an OAK-4 standalone app

From the repository root, use the same wheel and source fixture. Stage only
these example files and one wheel, so the app cannot import a hidden checkout.

```bash
mkdir -p /tmp/axial-oak4/wheels
cp examples/image_align_depth_correction/main.py /tmp/axial-oak4/
cp examples/image_align_depth_correction/requirements.txt /tmp/axial-oak4/
cp examples/image_align_depth_correction/oakapp.toml /tmp/axial-oak4/
cp /tmp/axial-dist/depthai_nodes-*.whl /tmp/axial-oak4/wheels/
python -m pip download --dest /tmp/axial-oak4/wheels --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 --platform linux_aarch64 --python-version 3.11 --implementation cp --abi cp311 --only-binary=:all: -r /tmp/axial-oak4/requirements.txt
oakctl app list -d DEVICE --format json
oakctl app run /tmp/axial-oak4 -d DEVICE
oakctl app logs 00000000-0000-0000-0000-000000000000 -d DEVICE --no-follow
```

Run the app in the foreground and wait for it to exit before reading retained
logs with the next command.

Use an authenticated, exclusively available device. Do not replace a running
development app. This one-shot app exits after writing its JSON result to logs
and `/tmp/axial-verification.json` inside its container. ImageAlign and
PointCloud use the device implementation. The correction executes on the
OAK-4 application CPU. The controller only builds, starts, and reads logs.
Dependency installation uses the staged arm64 wheels without device downloads.

Record the wheel SHA-256, source revision, device identity, app build identity,
and complete JSON output with the benchmark evidence. A passed synthetic
fixture does not replace cross-device snapshot replay, NeuralDepth rectification
metadata checks, or frozen reference-scene parity.

See the [API reference](../../docs/axial-depth-correction.md) for production
wiring and provenance handling.

# Support plane calibration on OAK4

Commission a fixed installation in a browser. The app sends typed controls and
verification requests to `SupportPlaneCalibration`. It shows the latest fit
with every assessment and lets the operator approve and store a passing plane.
The live depth overlay highlights points within the selected plane's
perpendicular tolerance. The fit and verification panels identify their own
source cloud sequence numbers.

## Run standalone

Use an OAK4 with calibrated depth cameras and an accelerometer. Position it so
the support surface is visible, then keep the mount fixed. The first container
build needs access to the base image and Python dependencies.

From the repository root, stage the current checkout as an OAK App:

```sh
stage_parent=$(mktemp -d)
sh examples/support_plane_calibration/stage.sh "$stage_parent/support-plane"
oakctl app run "$stage_parent/support-plane" -d <OAK4-IP>
```

`oakctl app run` uses the device's development app slot. Open
`http://<OAK4-IP>:8082` on your local network. The app serves its own HTML,
CSS, JavaScript, and same-origin JSON endpoints. Use it on a trusted network.

The staging script packages this checkout so the app can use the new node
before it is released on PyPI.

## Commission a plane

1. Click **Start search**. The app updates the fit panel with empty, rejected,
   or mixed batches until a plane passes or you click **Stop**.
1. Select a plane to highlight it on the live depth view. Click **Verify
   selected** to request a fresh assessment. A passing reply enables
   **Approve and save** after the search ends.
1. Click **Approve and save** to activate the equation and write the versioned
   record to the app's persistent storage.
1. After a restart, click **Verify and restore** to activate a compatible
   saved plane only after a fresh passing check.
1. If the camera moves or the installation changes, click **Invalidate** before
   commissioning it again. Set `INSTALLATION_ID` to name the physical setup.
   The example uses one persistent approval per standalone app.

## Try the browser workflow without hardware

Install the checkout with `python -m pip install -e '.[box-measurement]'`. From
the repository root, run:

```sh
python examples/support_plane_calibration/main.py --simulate \
    --height-range 700 900 \
    --store "$(mktemp -d)/support-plane.json"
```

Open [the local viewer](http://127.0.0.1:8082). The same DepthAI node runs
against a synthetic organized cloud with two surfaces and a synthetic IMU.
The optional height range accepts the 800 mm plane and rejects the 1100 mm plane.
Reuse the same `--store` path across restarts to test restoration.

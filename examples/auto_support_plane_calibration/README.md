# Automatic support plane on OAK4

Watch `AutoSupportPlane` pick a passing assessment, hold the plane, and check it
in continuous mode. The browser shows the live depth, held equation, latest
verification, and every assessment from the latest fit. During a refit, the
overlay continues to use the held plane until a replacement passes.

## Run standalone

Use an OAK4 with calibrated depth cameras and an accelerometer. From the
repository root, stage the current checkout and run the app:

```sh
stage_parent=$(mktemp -d)
sh examples/auto_support_plane_calibration/stage.sh "$stage_parent/auto-plane"
oakctl app run "$stage_parent/auto-plane" -d <OAK4-IP>
```

Open `http://<OAK4-IP>:8082` on a trusted local network. The app defaults to
continuous mode and starts searching as soon as the pipeline runs. It shares
the camera setup and browser server with the
[manual commissioning example](../support_plane_calibration/README.md).

## Watch a refit without hardware

Install the checkout with `python -m pip install -e '.[box-measurement]'`, then
run from the repository root:

```sh
python examples/auto_support_plane_calibration/main.py --simulate --port 8083
```

Open [the local viewer](http://127.0.0.1:8083). The node first picks the 800 mm
plane, which has more sampled support. Click **Set scene to 1100 mm** to replace
the synthetic cloud. After three failed checks, the node fits and holds the
1100 mm plane. Click **Set scene to 800 mm** to repeat the sequence.

Run with `--mode on_start` to keep the first passing plane without periodic
checks. Use `--initial-plane PATH` to verify a supplied `SupportPlane` JSON
equation at startup. A rejected initial plane triggers a search.

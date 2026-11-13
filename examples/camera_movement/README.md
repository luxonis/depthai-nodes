# OAK4 IMU desk recording

This capture gives the detector repeatable examples of a loose mount, a sudden
reorientation, vibration, and a magnetic disturbance. It uses only IMU reports.
The script marks where the app would request calibration. Put the camera in
its intended pose before starting the capture.

## Setup

- Put the OAK4 on a stable surface with slack in the cable. Mark its position
  and heading on paper or with tape. Mark a second heading about 20° away.
- Have a small wedge or folded cardboard ready to hold a roughly 10° downward
  tilt. Keep the camera supported during quick movements. A brisk hand turn
  is enough.
- Keep a steel tool nearby for the magnetic-disturbance cue. Move it near the
  camera and back without touching the camera or its support.
- Stop the application currently using the OAK4 before running the recorder.
  Use the terminal that launched it or the device's app manager.

## Guided capture

Run from the repository root in a Python environment with DepthAI 3.12.
Replace `OAK_IP` with the camera's IP address or omit `--device` for default
discovery. The default captures accelerometer and gyro.

```sh
oakctl run-script python examples/camera_movement/record_imu.py \
  --device OAK_IP --out examples/camera_movement/recordings/oak4-desk-01
```

Follow the printed cues. Each tilt and turn is followed by a hold at the new
pose, a return to the marked pose, and another still period. The full sequence
takes 2 minutes 37 seconds.

| Cue | Action |
| --- | --- |
| Baseline | Leave the camera still for 15 seconds. This is the only calibration marker in the take. |
| Slow and quick tilt | Tilt down about 10° over the slow window or in one brisk motion during the quick window. Keep the changed pose until the return cue. |
| Slow and quick yaw | Turn the camera about 20° while keeping its base flat. Keep the changed heading until the return cue. |
| Bump | Tap the desk near the camera a few times. Leave the camera in the marked pose. |
| Magnetic disturbance | Move the steel tool near the stationary camera and back away. |

A take produces `recording_*.tar` with native DepthAI IMU messages and
`imu.jsonl` with unique per-sensor samples, timestamps, and labeled phase
boundaries. The script prints sample counts. Keep both files and any notes
about missed cues or overshot marks. Choose a fresh output directory for each
take.

For a second guided take, add `--magnetometer-only` and change the output
directory to `oak4-magnetic-01`. The desk OAK4-D streams magnetic data on its
own. With DepthAI 3.12, requesting it together with accelerometer and gyro
produced no usable reports. The two takes can characterize rotation and
magnetic interference independently. Testing sensor fusion needs a simultaneous
recording.

A ten-minute stationary recording is useful for checking heading drift after
warm-up. Use the same command with
`--stationary-seconds 600` and a fresh output directory, such as
`oak4-stationary-01`. A second guided take in a fresh directory can be held back
to verify thresholds tuned on the first.

## Replay on the host

```sh
python examples/camera_movement/record_imu.py \
  --replay examples/camera_movement/recordings/oak4-desk-01/recording_*.tar
```

The replay check opens no device and prints the number of native IMU packets
read in three seconds.

## Use the detector

`CameraMovementDetector` accepts `dai.IMUData` with accelerometer and gyroscope
reports in the same frame. Configure it with the output of your IMU node.

```python
from depthai_nodes.message import CameraMovementControl
from depthai_nodes.node import CameraMovementDetector

detector = pipeline.create(CameraMovementDetector).build(imu.out)
control = detector.inputControl.createInputQueue()
events = detector.out.createOutputQueue()
status = detector.status.createOutputQueue()
pipeline.start()

# Send when the app knows the camera is at its intended pose.
control.send(CameraMovementControl.calibrate())
```

Wait for `baseline_ready` on `status`. Calibration uses fresh stationary samples
and replaces the reference only after success. Status reports the latest
calibration outcome, the last tilt estimate, and whether the IMU has stalled.
It runs once per second by default. Use `status_interval_s=None` to disable it.

The default tilt alert requires at least 3° of sustained tilt for 0.5 seconds.
It rearms after tilt stays at or below 1.5° for the same duration. The event
carries the angle, reason, and the IMU report timestamps.

To detect short horizontal turns, pass
`CameraMovementConfig(turn_threshold_deg=15)` to `build`. This emits
`turn_detected` for a completed turn around gravity. The estimate is limited
to the current turn and an IMU gap can interrupt it. A return turn can
produce its own event. Long-term heading needs a reliable heading reference.

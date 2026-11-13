# DepthAI Nodes

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![CI](https://github.com/luxonis/depthai-nodes/actions/workflows/ci.yaml/badge.svg?event=pull_request)
[![codecov](https://codecov.io/gh/luxonis/depthai-nodes/graph/badge.svg?token=ZG493MZ07B)](https://codecov.io/gh/luxonis/depthai-nodes)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)

<a name="overview"></a>

## 🌟 Overview

DepthAI Nodes provides reusable nodes and helpers for **DepthAI v3** pipelines, including neural network post-processing, image utilities, message handling, and runtime integrations. Inference helpers support both native DepthAI parsers and Python host parsers.

For complete Python API documentation, see the [DepthAINodes documentation and API reference](https://docs.luxonis.com/software-v3/ai-inference/inference/depthai-nodes/depthai-nodes-api-reference/).

## 📜 Table of Contents

- [🌟 Overview](#overview)
- [🛠️ Installation](#installation)
- [📦 Content](#-content)
  - [📨 Message](#-message)
  - [🧩 Node](#-node)
  - [Support plane calibration](#support-plane-calibration)
  - [Runtime](#runtime)
- [🤝 Contributing](#-contributing)

<a name="installation"></a>

## 🛠️ Installation

Install from PyPI:

```bash
pip install depthai-nodes
```

The box measurement and support plane nodes need open3d. Install it with the `box-measurement` extra:

```bash
pip install "depthai-nodes[box-measurement]"
```

On Debian or Ubuntu, open3d also needs system libraries that minimal images such as Docker containers and CI runners often lack:

```bash
sudo apt-get install libegl1 libgl1 libgomp1
```

open3d 0.20 publishes Linux wheels only for glibc 2.35 or newer, such as Debian 12+ or Ubuntu 22.04+. On older systems, including Debian 11, Ubuntu 20.04 and JetPack 5, pip cannot find a matching open3d and the install fails.

Or install from source:

```bash
git clone https://github.com/luxonis/depthai-nodes.git
cd depthai-nodes
pip install .
```

## 📦 Content

### 📨 Message

The `message` module provides `Collection`, `GatheredData`, and `SnapData`
messages, support plane messages such as `SupportPlane` and
`SupportPlaneCalibrationResult`, box measurement messages such as
`BoxDetections` and `BoxMeasurements`, plus creator functions for native DepthAI parser messages. Creators
cover detections, segmentation, classification, keypoints, maps, and other model
outputs. See the [message package documentation](./depthai_nodes/message/__init__.py) for the available
types and their roles.

### 🧩 Node

The `node` module provides parsers and pipeline helpers:

- **Parser nodes** handle model post-processing for architectures such as YOLO, MediaPipe, and YuNet.
- **Inference helpers**, including `ParsingNeuralNetwork` and `ParserGenerator`, create and connect inference and parser nodes.
- **Utility nodes** handle detection filtering, image overlays, colormaps, and message collection.
- **Support plane nodes** find the support plane in a point cloud, checked against gravity from IMU data. `AutoSupportPlane` finds and holds the support plane, and can optionally check and refit it. `SupportPlaneCalibrator` and `AutoSupportPlane` require open3d.
- **Box measurement nodes** detect, track, and measure boxes from color and a point cloud. `BoxDetector` and `BoxMeasurer` run on RVC4 only. `BoxDimensioner` requires open3d and has no RVC4-only check.

`ParsingNeuralNetwork` and `ParserGenerator` use native DepthAI parsers by default. Use `HostParsingNeuralNetwork` or pass `hostOnly=True` to `ParserGenerator.build()` to select this package's host parsers.

See the [node package documentation](./depthai_nodes/node/__init__.py) for available nodes and examples.

### Support plane calibration

Use `SupportPlaneCalibration` to commission a fixed camera. It requires the
`box-measurement` extra. Connect an organized point cloud in millimeters and an
accelerometer stream expressed in that cloud's coordinate frame.

```python
from depthai_nodes.message import (
    SupportPlaneCalibrationControl,
    SupportPlaneVerificationRequest,
)
from depthai_nodes.node import SupportPlaneCalibration, SupportPlaneCalibrationConfig

calibration = pipeline.create(SupportPlaneCalibration).build(
    pointCloud,
    imu,
    (width, height),
    config=SupportPlaneCalibrationConfig(vertical_distance_range_mm=(700, 900)),
)
control = calibration.inputControl.createInputQueue()
candidates = calibration.candidatesOutput.createOutputQueue(blocking=True)
verify = calibration.inputVerification.createInputQueue()
checks = calibration.verificationOutput.createOutputQueue(blocking=True)

pipeline.start()
control.send(SupportPlaneCalibrationControl.start())
while True:
    batch = candidates.get()
    show_assessments(batch.assessments)
    if any(assessment.accepted for assessment in batch.assessments):
        break

# The application presents the batch and obtains the reviewer's selection.
verify.send(SupportPlaneVerificationRequest(selected_plane, request_id="review-42"))
reply = checks.get()
```

Send `SupportPlaneCalibrationControl.stop()` to cancel a search. Send another
`start()` after cancellation, or if the reviewer declines the batch. The
verification reply echoes the request ID and contains a fresh assessment.
Each batch and reply carries the source cloud's Buffer metadata and stream ID.
The gravity direction and the accelerometer report's timestamps are available
separately as `gravity_direction`, `imu_timestamp_device`, `imu_timestamp_host`,
and `imu_timestamp_system`.

After explicit approval, save the plane against the device and installation
frame. Apply the application's resume policy to a compatible record on startup.

```python
from datetime import datetime, timezone

from depthai_nodes import ActiveSupportPlane, SupportPlaneStore, resolve_support_plane_path

store = SupportPlaneStore(resolve_support_plane_path("inspection", "line-4"))
store.save(ActiveSupportPlane(
    plane=approved_plane,
    device_id=device_id,
    frame_id=frame_id,
    approved_at=datetime.now(timezone.utc),
))
saved = store.load(device_id=device_id, frame_id=frame_id)
```

When the application withdraws approval, call `store.invalidate()` and pause
processing that depends on the plane.

The [standalone OAK4 example](examples/support_plane_calibration/README.md)
provides a browser workflow for search, review, verification, and approval.
The [automatic OAK4 example](examples/auto_support_plane_calibration/README.md)
shows automatic selection, periodic verification, and refitting with
`AutoSupportPlane`.

### Runtime

The `runtime` module contains runtime integrations. Its OAK4 QNN helper
creates ONNX Runtime sessions on the Hexagon DSP when used with the
`onnxruntime` variant of the `oakapp-base` image:

```python
from depthai_nodes.runtime import onnx_qnn_session

session = onnx_qnn_session("model.onnx")
```

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, parser guidelines, and testing instructions. Feedback and bug reports are welcome in [GitHub issues](https://github.com/luxonis/depthai-nodes/issues).

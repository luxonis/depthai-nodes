# DepthAI Nodes

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![CI](https://github.com/luxonis/depthai-nodes/actions/workflows/ci.yaml/badge.svg?event=pull_request)
[![codecov](https://codecov.io/gh/luxonis/depthai-nodes/graph/badge.svg?token=ZG493MZ07B)](https://codecov.io/gh/luxonis/depthai-nodes)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Overview

DepthAI Nodes is a Python "contrib" library designed to simplify host-side development with a growing collection of modular, high-level nodes. These cover a range of common needs - from neural network post-processing and I/O patterns to utility nodes for faster prototyping. With just a few lines of code, you can scaffold sophisticated pipelines, saving time and reducing boilerplate. In order to use these nodes you need to have your pipeline written with `DepthAIv3`.

**NOTE**:
We are always listening to the community so feel free to report and feedback, issues or contribute to the library with our own host nodes.

## 📜 Table of Contents

- [🌟 Overview](#overview)
- [🛠️ Installation](#-installation)
- [📦 Content](#-content)
  - [📨 Message](#-message)
  - [🧩 Node](#-node)
- [🤝 Contributing](#-contributing)

## 🛠️ Installation

The `depthai_nodes` package is hosted on PyPI, so you can install it with `pip`.

To install the package, run:

```bash
pip install depthai-nodes
```

### Manual installation

If you want to manually install the package from the GitHub repository you can run:

```bash
git clone git@github.com:luxonis/depthai-nodes.git
```

and then inside the directory run:

```bash
pip install .
```

## 📦 Content

This library is organized into two primary modules, each focused on a specific aspect of working with DepthAI on the host side:

- `message` - Custom message types
- `node` - High-level, modular host-side nodes

### 📨 Message

The `message` module defines a set of extended message types designed to simplify working with outputs from various neural networks. These go beyond the standard DepthAI messages and include richer data structures for tasks such as object detection, segmentation, classification, pose estimation, and more.

These enhanced messages aim to reduce the boilerplate code needed for parsing and interpreting NN outputs, making it easier to plug them into visualization or processing pipelines. You can learn more about each message type in the dedicated [README](./depthai_nodes/message/README.md).

### 🧩 Node

The `node` module provides a collection of ready-to-use host-side nodes that abstract common processing patterns and tasks. These nodes fall into three main categories:

- **Parser nodes** - Handle post-processing for specific model architectures such as YOLO, MediaPipe, YuNet, etc.
- **Helper nodes** - Like ParsingNeuralNetwork and ParserGenerator which help manage simple or complex model outputs more efficiently.
- **Utility nodes** – Perform common operations like detection filtering, drawing overlays, applying segmentation colormaps, and more—all in just a few lines of code.

This modular approach allows you to rapidly prototype and scale complex applications with less effort while keeping your code clean and maintainable.

To read more about the nodes and see simple examples, please refer to the [nodes documentation](./depthai_nodes/node/README.md).

## 🤝 Contributing

If you want to contribute to this project, read the instructions in [CONTRIBUTING.md](./CONTRIBUTING.md)

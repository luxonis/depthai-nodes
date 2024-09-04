# DepthAI Nodes

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![CI](https://github.com/luxonis/depthai-nodes/actions/workflows/ci.yaml/badge.svg)
![Coverage](https://github.com/luxonis/depthai-nodes/blob/dev/media/coverage_badge.svg)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DepthAI Nodes package includes parser nodes for decoding and postprocessing outputs from Neural Network node in DepthAI.

**NOTE**:
The project is in an alpha state, so it may be missing some critical features or contain bugs - please report any feedback!

## Table of Contents

- [Installation](#installation)
- [Contributing](#contributing)

## Installation

The `depthai_nodes` package requires Python 3.8 or later and `depthai v3` installed.
While the `depthai v3` is not yet released on PyPI, you can install it with the following command:

```bash
pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-release-local/ depthai==3.0.0a2
```

The `depthai_nodes` package is hosted on PyPI, so you can install it with `pip`.

To install the package, run:

```bash
pip install depthai-nodes
```

### Manual installation

If you want to manually install the package from GitHub repositoory you can run:

```bash
git clone git@github.com:luxonis/depthai-nodes.git
```

and then inside the directory run:

```bash
pip install .
```

Note: You'll still need to manually install `depthai v3`.

## Contributing

If you want to contribute to this project, read the instructions in [CONTRIBUTING.md](./CONTRIBUTING.md)

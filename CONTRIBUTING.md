# Contributing to DepthAI Nodes

This guide covers setting up a development environment, adding parsers, and validating changes before opening a pull request.

## Development setup

Clone the repository and run these commands from its root, preferably in a virtual environment:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
pre-commit install
```

Pre-commit runs linting and formatting checks on each commit. Some hooks modify files in place; review and stage those changes before committing again. To check the whole repository:

```bash
pre-commit run --all-files
```

## Developing parsers

Before adding a parser, check whether an existing parser supports the model or can be extended to support it. Follow the structure and naming conventions in [existing parsers](depthai_nodes/node/parsers).

- Implement the node using `BaseParser`, with configuration handled by `build()` and tensor processing separated into helpers in `depthai_nodes/node/parsers/utils`.
- Reuse existing output messages and [message creators](depthai_nodes/message/creators). Add a creator when a new message conversion is needed.
- Preserve input timestamps and transformation metadata when creating output messages.
- Export the parser from both `depthai_nodes/node/parsers/__init__.py` and `depthai_nodes/node/__init__.py`, including their `__all__` lists. Update `ParserGenerator` when the parser should be selected from NN Archive metadata.
- Add tests for the parser's computations, configuration, and output messages. Update the parser catalogue in the [node package overview](depthai_nodes/node/__init__.py).

### NN Archive configuration

Use the exact NN Archive keys when reading head configuration in `build()`. For example, read `n_classes`, not `num_classes`. If the parser uses a different internal attribute name, map the archive key explicitly, as parsers do for `classes` and `label_names`.

Common configuration keys include:

| Key              | Meaning                                                          |
| ---------------- | ---------------------------------------------------------------- |
| `outputs`        | Names of the output tensors consumed by the head.                |
| `classes`        | Class names in the model's class order.                          |
| `n_classes`      | Number of classes.                                               |
| `conf_threshold` | Confidence threshold for detections.                             |
| `iou_threshold`  | Intersection-over-union threshold for non-maximum suppression.   |
| `max_det`        | Maximum number of detections per image.                          |
| `anchors`        | Anchor box dimensions grouped by model output.                   |
| `is_softmax`     | Whether the model output already contains softmax probabilities. |
| `subtype`        | Model variant used to select decoding logic.                     |
| `n_prototypes`   | Number of mask prototypes for instance segmentation.             |
| `n_keypoints`    | Number of keypoints per detected instance.                       |

The required keys and tensor layouts depend on the model and parser. Check the model's archive and the relevant parser's `build()` implementation rather than assuming every parser accepts every key.

## Documentation

Use **Google-style docstrings**. Keep package overviews and examples in package `__init__.py` docstrings, and update them when public behavior changes.

Build the API reference from the repository root:

```bash
pydoctor depthai_nodes
```

Review `apidocs/index.html`. CI runs the same build with warnings treated as errors.

## Testing

Add or update tests for changed behavior. The repository has three test suites:

| Suite                                       | Coverage                                                          | Requirements                                            |
| ------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------- |
| Unit tests (`tests/unittests`)              | Individual components, message creators, and parser computations. | No device required.                                     |
| Integration tests (`tests/stability_tests`) | Parsing recorded `NNData` and checking the resulting messages.    | Access to the test-data bucket; no device required.     |
| End-to-end tests (`tests/end_to_end`)       | Complete inference pipelines on real devices.                     | OAK hardware and HubAI credentials for model downloads. |

### Unit tests

Run from the repository root:

```bash
pytest tests/unittests
```

### Integration tests

Ask the code owners for test-data bucket credentials, then set them in your environment:

```bash
export B2_APPLICATION_KEY="your_application_key"
export B2_APPLICATION_KEY_ID="your_application_key_id"
```

Run from `tests/stability_tests`:

```bash
python main.py -all --download --duration 2
```

This tests each parser for two seconds. Adjust `--duration` as needed, or select a subset with `--parser` or `--model`.

### End-to-end tests

Set the device IP addresses and HubAI credentials:

```bash
export RVC2_IP="your_rvc2_ip"
export RVC4_IP="your_rvc4_ip"
export HUBAI_TEAM_SLUG="your_hubai_team_slug"
export HUBAI_API_KEY="your_hubai_api_key"
```

Run from `tests/end_to_end`:

```bash
python main.py -all
```

Use `--model` to select models, `--platform` to select the device platform, and `--depthai-nodes-version` to select the model set associated with a branch or release. Run `python main.py --help` for available options.

## Pull requests and review

1. Make changes on a new branch.
1. Run the relevant tests and pre-commit checks. Build the API reference if you changed docstrings or public exports.
1. Open a pull request describing the change and how you tested it.
1. Request review from the maintainers:
   - [Klemen Škrlj](https://github.com/klemen1999)
1. Add other relevant team members as reviewers and address review feedback before the team merges the pull request.

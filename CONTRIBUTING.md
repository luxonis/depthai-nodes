# Contributing to DepthAI Nodes

It outlines our workflow and standards for contributing to this project.

## Table of Contents

- [Developing parser](#developing-parser)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Documentation](#documentation)
  - [Editor Support](#editor-support)
- [Testing](#testing)
- [Making and Reviewing Changes](#making-and-reviewing-changes)

## Developing parser

Parser should be developed so that it is consistent with other parsers. Check out other parsers to see the required structure. Additionally, pay attention to the naming of the parser's attributes. Check out [Developer guide](docs/developer_guide.md).

**NOTE:** When adding new features make sure that you extend the tests appropriately. Check out the [Testing](#testing) section for more information.

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency:

1. Install pre-commit (see [pre-commit.com](https://pre-commit.com/#install)).
1. Clone the repository and run `pre-commit install` in the root directory.
1. The pre-commit hook will now run automatically on `git commit`.
   - If the hook fails, it will print an error message and abort the commit.
   - It will also modify the files in-place to fix any issues it can.

## Documentation

We use the [Epytext](https://epydoc.sourceforge.net/epytext.html) markup language for documentation.
To verify that your documentation is formatted correctly, follow these steps:

1. Download [`get-docs.py`](https://github.com/luxonis/python-api-analyzer-to-json/blob/main/gen-docs.py) script
1. Run `python3 get-docs.py luxonis_ml` in the root directory.
   - If the script runs successfully and produces `docs.json` file, your documentation is formatted correctly.
   - **NOTE:** If the script fails, it might not give the specific error message. In that case, you can run
     the script for each file individually until you find the one that is causing the error.

### Editor Support

- **PyCharm** - built in support for generating `epytext` docstrings
- **Visual Studio Code** - [AI Docify](https://marketplace.visualstudio.com/items?itemName=AIC.docify) extension offers support for `epytext`
- **NeoVim** - [vim-python-docstring](https://github.com/pixelneo/vim-python-docstring) supports `epytext` style

## Testing

We have 3 types of tests:

- Unit tests (`tests/unittests`)
- Integration tests (`tests/stability_tests`)
- End-to-end tests (`tests/end_to_end`)

All tests are located in the `tests` directory, each in its own subdirectory. Unit tests and integration tests are running on every PR while end-to-end tests are triggered on every push to the main branch, or when manually triggered. In the unit tests we check individual components (messages, parser's functions, etc.). In the integration tests we check if the parser is able to parse the output of the neural network, we have predefined `NNData` for each parser and expected output message and we check if the output message is correct. Lastly, in the end-to-end tests we check if the parser is running in the complete pipeline with camera, neural network, and parsers on real device.

While end-to-end tests require real device, integration tests and unit tests can be run without it.

### Running unit tests

To run unit tests, first install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Then run the tests from the root directory:

```bash
pytest tests
```

### Running integration tests

To run integration tests, first install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Then you will need access to our bucket to download the predefined `NNData` for each parser. You can get the credentials from the code owners. Export the credentials as environment variables:

```bash
export B2_APPLICATION_KEY=<your_application_key>
export B2_APPLICATION_KEY_ID=<your_application_key_id>
```

Then run the tests from the `tests/stability_tests` directory:

```bash
python main.py -all --download --duration 2
```

This will run the integration tests for two seconds for each parser. You can specify the duration with `--duration` flag.

### Running end-to-end tests

To run end-to-end tests, first install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

You will also need to specify the IP addresses of the RVC2 and RVC4 devices. To get the models you want to test on, you will need HubAI credentials.

```bash
export RVC2_IP=<your_rvc2_ip>
export RVC4_IP=<your_rvc4_ip>
export HUBAI_TEAM_SLUG=<your_hubai_team_slug>
export HUBAI_API_KEY=<your_hubai_api_key>
```

Then run the tests from the `tests/end_to_end` directory:

```bash
python main.py -all
```

This will run the end-to-end tests for all public models. You can specify the models with `--model` flag, `--platform` flag to specify the platform to test on and `--depthai-nodes-version` to specify the version of depthai-nodes to test on.

## Making and Reviewing Changes

1. Make changes in a new branch.
1. Test your changes locally.
1. Commit (pre-commit hook will run).
1. Push to your branch and create a pull request. Always request a review from:
   - [Matija Teršek](https://github.com/tersekmatija)
   - [Jakob Mraz](https://github.com/jkbmrz)
   - [Jaša Kerec](https://github.com/kkeroo)
   - [Klemen Škrlj](https://github.com/klemen1999)
1. Any other relevant team members can be added as reviewers as well.
1. The team will review and merge your PR.

# End-to-end tests

This directory contains end-to-end tests for DepthAI. These tests are designed to run on a real device and test the complete pipeline with camera, neural network, and parsers.
The tests check if the device is able to run a model with neural network node and parse the output with parser node and return the message. If the error is raised during the process, the test will fail.

## Running the tests on GitHub Actions

You can manually trigger the Github action to run the HIL tests. There is a workflow parameter `additional-parameter` that can be used to specify the desired test. Default is `-all`. The available options are: `-all`, `-p <parser_name>`, `-m <model_1> <model_2> ...`, `-nn <path_to_archive_1> <path_to_archive_2> ...`. You can also specify with branch you want to run the tests on. The default is `main`. NOTE: There is a custom parameter `branch` and not the built-in one.

### Required parameters

There are 4 required parameters that need to be set when triggering the Github action:

- `additional-parameter`: The parameter that specifies the desired test. Default is `-all` which runs tests on all public models. The available options are: `-all`, `-p <parser_name>`, `-m <model_1> <model_2> ...`.
- `branch`: The branch on which the tests will be run. Default is `main`.
- `testbed`: The testbed on which the tests will be run. Default is `oak4-pro`. Available: `oak4-pro`, `oak4-s`.
- `depthai-version`: The version of the DepthAI that will be used for the tests. Default is `3.0.0a6`.

## Running the tests locally

Currently, you must specify the device IP address in the ENV variables: `RVC2_IP` and `RVC4_IP`. If the ENV variable is empty the script will take the connected device via USB. For sending requests to HubAI you also need to specify `HUBAI_TEAM_ID` and `HUBAI_API_KEY` ENV variables.
For running the tests locally you can use `main.py` script. You can specify the models from ZOO or specify the path to the local NN archive paths. If you want to test all available models you can use `--all` flag and for testing specific parser on all models you can use `--parser` or `-p` flag.

Test all public models from the ZOO:

```bash
python main.py --all
```

Test specific models from the ZOO:

```bash
python main.py -m <model_1> <model_2> ...
```

Test local NN archives:

```bash
python main.py -nn <path_to_archive_1> <path_to_archive_2> ...
```

Test specific parser on all models:

```bash
python main.py -p <parser_name>
```

You can also run `manual.py` with `-m` or `-nn` if want to debug parser quickly (without pytest) and by specifying `-ip` you can connect to the specifi device with IP or mxid.

## Limitations

Some models are not yet fully supported because either DAI can not produce required camera output (small input size, unsupported format) or the model requires multiple cameras.

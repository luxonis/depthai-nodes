# End-to-end tests

This directory contains end-to-end tests for DepthAI. These tests are designed to run on a real device and test the complete pipeline with camera, neural network, and parsers.
The tests check if the device is able to run a model with neural network node and parse the output with parser node and return the message. If the error is raised during the process, the test will fail.

## Running the tests

Currently, you must specify the device IP address in the ENV variables: `RVC2_IP` and `RVC4_IP`. If the ENV variable is empty the script will take the connected device via USB.
For running the tests locally you can use `manual.py` script. You can specify the model slugs to test the models from ZOO or specify the path to the local NN archive paths. If you want to test all available models you can use `--all` flag and for testing specific parser on an models you can use `--parser` or `-p` flag.

Test all public models on ZOO:

```bash
python manual.py --all
```

Test specific models on ZOO given the slugs:

```bash
python manual.py -s <slug_1> <slug_2> ...
```

Test local NN archives:

```bash
python manual.py -nn <path_to_archive_1> <path_to_archive_2> ...
```

Test specific parser on all models:

```bash
python manual.py -p <parser_name>
```

## Limitations

- The test does not yet connect to the testbed. It will be added soon.
- The test does not yet get all the models from HubAI via API call. It will be added soon.

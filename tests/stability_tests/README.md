# Stability tests for parsers and host nodes in Depthai-nodes

## Overview

This directory contains stability tests for parsers and host nodes in DepthAI-nodes. These tests are designed to simulate a neural network output and send it to the parser node. The parser node will parse the output and return the message. The tests check if the parser outputs the expected message. Since we know what `NNData` we are sending to the parser, we can check if the outputed message is correct.

The stability tests runs without the device. It uses the mock pipeline and mock queues to send the `NNData` to the parser node and check the output.

The tests for parsers pipeline are in this directory while the stability tests for host nodes re-use their unit tests by running them for specified amount of time.

The testing pipeline is as follows:

1. Load the neural network archive.
1. Load the parser node with `ParserGenerator` based on the NN archive.
1. Read and load the `NNData` from pickle file. It contains the output of the neural network.
1. Send the `NNData` to the parser node.
1. Read and load the expected message from the pickle file.
1. Check if the parser node outputs the expected message.

We are storing tests in the B2 bucket. In the beginning, we download the tests from the bucket and store them in the `nn_datas` directory. The tests are stored in the following structure:

```
nn_datas
├── <parser_name>
│   ├── <model.pkl> # Contains the NNData
│   └── <model_output.pkl> # Contains the expected message
│   └── <model.png> # Contains the input image
```

for example:

```
nn_datas
├── ClassificationParser
│   ├── efficientnet-lite.pkl
│   └── efficientnet-lite_output.pkl
│   └── efficientnet-lite.png
├── FastSAMParser
│   ├── fastsam-s.pkl
│   └── fastsam-s_output.pkl
│   └── fastsam-s.png
```

## Test generation

To generate a new test for the parser, you can use `extract_nn_data.py` script. The script will extract the `NNData` from the neural network output and store it in the pickle file. The script requires the following arguments: `-m` for the model, `-img` for the input image, and optional `-ip` for the device IP or mxid.

The script does not generate the expected message because each parser has its own message format and DAI messages can not be dumped in the pickle file.

One example for generating the expected message for the `ClassificationParser`:

```python
expected_output = {
    'model': 'luxonis/efficientnet-lite:lite0-224x224',
    'parser': 'ClassificationParser',
    'classes': ['wolf', 'dog', 'cat'],
    'scores': [0.7, 0.2, 0.1]
}

with open('nn_datas/ClassificationParser/efficientnet-lite_output.pkl', 'wb') as f:
    pickle.dump(expected_output, f)
```

In the end, you should have all the files in the parser-specific directory inside `nn_datas` directory. You need to upload the parser directory to the B2 bucket.

## Running the tests locally

To run the tests, you can use the `main.py` script. You can use `--all` flag to test all parsers or test a specific parser with `-p` flag.
You would need the B2 credentials to download the tests from the bucket and set it in the ENV variables `B2_APPLICATION_KEY_ID` and `B2_APPLICATION_KEY`.

## Running the tests in the CI

The stability tests are triggered in every PR. But you can also trigger them manually. Required parameters are:

- `additional-parameter`: The parameter that specifies the desired test. Default is `-all` which runs tests on all parsers. The available options are: `-all`, `-p <parser_name>`.
- `depthai-version`: The version of the DepthAI that will be used for the tests. Default is `3.0.0a14`.
- `duration`: The duration of each test in seconds. Default is `10` seconds.

## How to create test for Host node or parser

### Parser

The process is quite simple. You need to generate the test for the parser node (see [Test generation](#test-generation)). Then add the entry in the `check_messages.py` file in the `check_output` function and optionally create a new function to check the specific message, if needed. Last step is to add the model information to the `config.py` file so the tests know which model to download from the ZOO and set up the parser node.

### Host node

We re-use the unit tests for the host nodes. The testing differs between the nodes that inherit from `dai.Node.ThreadedHostNode` and the ones that inherit from `dai.Node.HostNode`. The `ThreadedHostNode` are tested similarly to the parsers by creating `InfiniteInput` which sends the message to the node for `duration` of seconds. If we dont specify the duration, then we run the test only once.

The nodes that inherit from `dai.Node.HostNode` are tested by creating the node and running the test for `duration` of seconds. If we dont specify the duration, then we run the test only once.

You dont need to manually create the mock classes since they are implemented in the `conftest.py` file. Potentialy just add methods or attributes to the mock classes.

Check the already implemented tests for reference.

- `ThreadedHostNode` - tests for `GatherData`
- `HostNode` - tests for `TilesPatcher`

You can check if everything works by running the tests locally. To run the unit tests move to the `depthai_nodes/tests` directory and run the tests with `pytest`.
To run the stability tests move to the `depthai_nodes/tests/stability_tests` directory and run the tests with `python main.py -all --duration 2`.

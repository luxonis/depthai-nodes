# Integration tests for Parsers in Depthai-nodes

## Overview

This directory contains integration tests for parsers in DepthAI-nodes. These tests are designed to simulate a neural network output and send it to the parser node. The parser node will parse the output and return the message. The tests check if the parser outputs the expected message. Since we know what `NNData` we are sending to the parser, we can check if the outputed message is correct.

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
│   ├── <model-slug.pkl> # Contains the NNData
│   └── <model-slug_output.pkl> # Contains the expected message
│   └── <model-slug.png> # Contains the input image
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

To generate a new test for the parser, you can use `extract_nn_data.py` script. The script will extract the `NNData` from the neural network output and store it in the pickle file. The script requires the following arguments: `-m` for the model slug, `-img` for the input image, and optional `-ip` for the device IP or mxid.

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

The integration tests are triggered in every PR. But you can also trigger them manually. Required parameters are:

- `additional-parameter`: The parameter that specifies the desired test. Default is `-all` which runs tests on all parsers. The available options are: `-all`, `-p <parser_name>`.
- `branch`: The branch on which the tests will be run. Default is `main`.
- `testbed`: The testbed on which the tests will be run. Default is `oak4-s`. Available: `oak4-pro`, `oak4-s`.
- `depthai-version`: The version of the DepthAI that will be used for the tests. Default is `3.0.0a6`.

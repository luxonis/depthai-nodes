# Nodes

## Table of Contents

- [Parser Nodes](#parser-nodes)
  - [Object Detection](#object-detection)
  - [Classification](#classification)
  - [Segmentation](#segmentation)
  - [Keypoints](#keypoints)
  - [Feature Matching](#feature-matching)
  - [Other](#other)
- [Utility & Helper Nodes](#utility--helper-nodes)
  - [Base Classes](#base-classes)
  - [Image Processing Nodes](#image-processing-nodes)
  - [Neural Network Processing](#neural-network-processing)
  - [Detection and Filtering](#detection-and-filtering)
  - [Data Management](#data-management)
- [Usage](#usage)
  - [Example](#example)

## Parser Nodes

Parser nodes are used to parse the output of a neural network. The main purpose of these nodes is to hide all postprocessing logic from the user. The node will send out a message with the parsed data (e.g. detections, keypoints, etc.).

### Object Detection

- `YOLOExtendedParser`: Extended YOLO detection parser that supports all YOLO models and tasks (detection, pose estimation, instance segmentation). It will output the [`depthai.ImgDetections`](https://docs.luxonis.com/software-v3/depthai/depthai-components/messages/img_detections) message with the detections.
- `YuNetParser`: YuNet face detection parser that will output the [`depthai.ImgDetections`](https://docs.luxonis.com/software-v3/depthai/depthai-components/messages/img_detections) message with the detections.
- `SCRFDParser`: SCRFD parser for parsing family of SCRFD models that will output the [`depthai.ImgDetections`](https://docs.luxonis.com/software-v3/depthai/depthai-components/messages/img_detections) message with the detections.
- `MPPalmDetectionParser`: MediaPipe palm detection parser that will output the [`depthai.ImgDetections`](https://docs.luxonis.com/software-v3/depthai/depthai-components/messages/img_detections) message with the detections.
- `PPTextDetectionParser`: Parser for parsing the output of the Paddle Text Detection model. It will output the [`depthai.ImgDetections`](https://docs.luxonis.com/software-v3/depthai/depthai-components/messages/img_detections) message with the detections.

### Classification

- `ClassificationParser`: General classification parser for postprocessing the output of a classification model and outputting the [`depthai_nodes.message.Classifications`](../message/README.md#classifications) message.
- `ClassificationSequenceParser`: Parser for models that predict the classes multiple times and return a list of predicted classes, where each item corresponds to the relative step in the sequence. In addition to time series classification, this parser can also be used for text recognition models where words can be interpreted as a sequence of characters (classes). It will output the [`depthai_nodes.message.Classifications`](../message/README.md#classifications) message.

### Segmentation

- `SegmentationParser`: Parser for parsing the output of the segmentation models. It will output the [`depthai_nodes.message.SegmentationMask`](../message/README.md#segmentationmask) message.
- `FastSAMParser`: Parser for parsing the output of the FastSAM model. It will output the [`depthai_nodes.message.SegmentationMask`](../message/README.md#segmentationmask) message.

### Keypoints

- `KeypointParser`: General keypoint parser that will output the [`depthai_nodes.message.Keypoints`](../message/README.md#keypoints) message.
- `SuperAnimalParser`: Special parser for parsing the output of the SuperAnimal model and will output the [`depthai_nodes.message.Keypoints`](../message/README.md#keypoints) message.
- `HRNetParser`: Special parser for parsing the output of the HRNet model and will output the [`depthai_nodes.message.Keypoints`](../message/README.md#keypoints) message.

### Feature Matching

- `XFeatMonoParser`: Special parser for parsing the output of the XFeat model when running in mono mode. It will output the `dai.TrackedFeatures` message.
- `XFeatStereoParser`: Special parser for parsing the output of the XFeat model when running in stereo mode. It will output the `dai.TrackedFeatures` message.

### Other

- `LaneDetectionParser`: Special parser for parsing the output of the Ultra-Fast-Lane-Detection model. It will output the [`depthai_nodes.message.Clusters`](../message/README.md#clusters) message.
- `MLSDParser`: Special parser for parsing the output of the MLSD model. It will output the [`depthai_nodes.message.Lines`](../message/README.md#lines) message.
- `EmbeddingsParser`: Simple parser that will only forward the output of the neural network. It will output the `dai.NNData` message.
- `RegressionParser`: Special parser for parsing the output of the regression models. It will output the [`depthai_nodes.message.Predictions`](../message/README.md#predictions) message.
- `MapOutputParser`: Special parser for parsing the output of the model that produces a map (like depth estimation). It will output the [`depthai_nodes.message.Map2D`](../message/README.md#map2d) message.
- `ImageOutputParser`: Special parser for parsing the output of the model that produces an image (like super-resolution models). It will output the `dai.ImgFrame` message.

## Utility & Helper Nodes

### Base Classes

- `BaseHostNode`: Abstract base class for all host nodes, providing common functionality and platform-specific configurations.

### Image Processing Nodes

- `ApplyColormap`: Applies a colormap to generic 2D arrays (maps/masks). Sends out `dai.ImgFrame` message.
- `ApplyDepthColormap`: Applies a colormap to depth/disparity `dai.ImgFrame` using percentile-based normalization for more stable depth visualization. Invalid depth values (\<= 0) are ignored when computing percentiles and rendered black. Sends out `dai.ImgFrame` message.
- `DepthMerger`: Merges detections with depth information. Sends out `dai.SpatialImgDetections` message.
- `ImgFrameOverlay`: A host node that receives two dai.ImgFrame objects and overlays them into a single one. Sends out `dai.ImgFrame` message.
- `Tiling`: Manages tiling of input frames for neural network processing, divides frames into overlapping tiles based on configuration parameters, and creates ImgFrames for each tile to be sent to a neural network node. Sends out `dai.ImgFrame` message.
- `TilesPatcher`: Handles the processing of tiled frames from neural network (NN) outputs, maps the detections from tiles back into the global frame, and sends out the combined detections for further processing. Sends out `dai.ImgDetections` message.

### Neural Network Processing

- `ParsingNeuralNetwork`: Node for creating a neural network node and relevant parser(s) for the given model from our Model ZOO. Does not send out any messages.
- `HostParsingNeuralNetwork`: Host-side `ParsingNeuralNetwork` implementation. Does not send out any messages.
- `ParserGenerator`: Generates parsers from the given NN archive. Does not send out any messages.

### Detection and Filtering

- `ImgDetectionsBridge`: Transforms the dai.ImgDetections to ImgDetectionsExtended object or vice versa. Note that conversion from ImgDetectionsExtended to ImgDetection loses information about segmentation, keypoints and rotation. Sends out `dai.ImgDetections` or [`depthai_nodes.message.ImgDetectionsExtended`](../message/README.md#imgdetectionsextended) message.
- `ImgDetectionsFilter`: Filters image detections based on various criteria. Sends out `dai.ImgDetections` or [`depthai_nodes.message.ImgDetectionsExtended`](../message/README.md#imgdetectionsextended) or `dai.SpatialImgDetections` message.
- `InstanceToSemanticMask`: Converts instance-id masks into semantic masks by mapping instance indices to detection class labels. Sends out `ImgDetectionsExtended`.

### Data Management

- `GatherData`: A node for gathering data. Gathers n messages based on reference_data. To determine n, wait_count_fn function is used. The default wait_count_fn function is waiting for len(TReference.detection). This means the node works out-of-the-box with dai.ImgDetections and ImgDetectionsExtended. Sends out `depthai_nodes.message.GatheredData` message.

## Usage

These nodes are designed to be used within DepthAI V3 pipelines to process and manipulate data from OAK devices. Each node provides specific functionality and can be combined to create complex processing pipelines.

For detailed usage examples and specific node configurations, please refer to the individual node documentation.

### Example

The entry point for using neural networks is usually the `ParsingNeuralNetwork` node, which accepts a model reference from Model ZOO (or the NN archive) and creates everything needed to run the model in the pipeline. You can read more about the NN archive in our documentation [here](https://docs.luxonis.com/software-v3/ai-inference/nn-archive/).

Example usage of `ParsingNeuralNetwork` node:

```python
from depthai_nodes.node import ParsingNeuralNetwork

camera_node = pipeline.create(dai.node.Camera).build()

nn = pipeline.create(ParsingNeuralNetwork).build(
    camera_node, model_source="luxonis/mediapipe-selfie-segmentation:256x144"
)
```

As `model_source` you can provide local NN Archive, model reference from Model ZOO or `dai.NNModelDescription` object.

This code section creates the `dai.NeuralNetwork` and `SegmentationParser` nodes required for postprocessing the results. Additionally, the `ParsingNeuralNetwork` node handles all the necessary connections: It connects the `Camera` node to the `NeuralNetwork` node, the `NeuralNetwork` node to the `SegmentationParser` node, and passes the `SegmentationParser` output to the `nn.out`.

Similarly, you can create any other utility, helper or parser node. For example, if you want to filter out the detections based on the label, you can use the `ImgDetectionsFilter` node.

```python
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsFilter

camera_node = pipeline.create(dai.node.Camera).build()

nn = pipeline.create(ParsingNeuralNetwork).build(
    camera_node, model_source="luxonis/yolov6-nano:r2-coco-512x288"
)

filter_node = pipeline.create(ImgDetectionsFilter).build(nn.out, labels_to_keep=[0])
```

This code section will pass the detections from the `nn.out` to the `filter_node` and filter out the detections with label `0`.

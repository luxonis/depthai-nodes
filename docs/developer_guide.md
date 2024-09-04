# Developer Guide

This guide is intended for developers who want to create new parsers for DepthAI. It explains the process of creating a new parser and the requirements for the parser to be compatible with DepthAI.

### Developing parser

Check the already implemented parsers to see the required structure. The parser should be developed so that it is consistent with other parsers. Additionally, pay attention to the naming of the parser's attributes. Check out [NN Archive Parameters](#nn-archive-params). Before, creating a new parser, check if there is already a parser for the model you want to implement. Check also if any parser can be slightly modified to support the new model. All parsers also sends out the results in form of messages.

### Messages

The parser should send out the results in form of messages. Before creating a new message, check if there is already a message that can be reused. You should also implement the message creator function which takes the models output and creates the message.

### Documenting the parser

The parser should have consistent documentation. When opening a pull request, the documentation is automatically generated and checked for consistency.

### NN Archive params

> The idea behind this is that the creator of the parser can accidentally name some parser's parameters slightly differently as they are named in the NN archive (num_classes instead of n_classes). Subsequently, DAI will not map the correct parameters.

Below are listed all the parameters supported in the NN Archive. Each parameter also has a computer vision task where it is required. You can help with the list to better plan and develop new parsers. E.g. if you are adding a classification parser many parameters are already present in NN Archive (n_classes, classes, is_softmax). You should reuse the naming in your parser’s code so DepthAI can automatically map the NN Archive parameters to the Parser.

- classes - Names of object classes detected by the model. `Object detection` `Classification` `Segmentation`
- n_classes - Number of object classes detected by the model. `Object detection` `Classification` `Segmentation`
- iou_threshold - Non-max suppression threshold limiting boxes intersection. `Object detection`
- conf_threshold - Confidence score threshold above which a detected object is considered valid. `Object detection`
- max_det - Maximum detections per image. `Object detection`
- anchors - Predefined bounding boxes of different sizes and aspect ratios. The innermost lists are length 2 tuples of box sizes. The middle lists are anchors for each output. The outmost lists go from smallest to largest output. `Object detection`
- is_softmax - True, if output is already softmaxed. `Classification` `Segmentation` `YOLO`
- yolo_outputs - A list of output names for each of the different YOLO grid sizes. `YOLO`
- mask_outputs - A list of output names for each mask output. `YOLO`
- protos_outputs - Output name for the protos. `YOLO`
- keypoints_outputs - A list of output names for the keypoints. `YOLO`
- angles_outputs - A list of output names for the angles. `YOLO`
- subtype - YOLO family decoding subtype (e.g. yolov5, yolov6, yolov7 etc.) `YOLO`
- n_prototypes - Number of prototypes per bbox in YOLO instance segmnetation. `YOLO`
- n_keypoints - Number of keypoints per bbox in YOLO keypoint detection. `YOLO`

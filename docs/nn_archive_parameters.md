# NN Archive params

> The idea behind this document is that the creator of the parser can accidentally name some parser's parameters slightly differently as they are named in the NN archive (num_classes instead of n_classes). Subsequently, DAI will not map the correct parameters.

Below are listed all the parameters supported in the NN Archive. Each parameter also has a computer vision task where it is required. You can help with the list to better plan and develop new parsers. E.g. if you are adding a classification parser many parameters are already present in NN Archive (n_classes, classes, is_softmax). You should reuse the naming in your parser’s code so DepthAI can automatically map the NN Archive parameters to the Parser.

### All parameters

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

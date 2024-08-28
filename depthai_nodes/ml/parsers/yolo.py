import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message
from .utils.yolo import decode_yolo_output, process_single_mask, parse_kpts


KPTS_MODE = 0
SEG_MODE = 1


class YOLOParser(dai.node.ThreadedHostNode):
    def __init__(
            self,
            confidence_threshold: int = 0.5,
            num_classes: int = 1,
            iou_threshold: int = 0.5
        ):
        """Initialize the YOLOParser node.

        @param confidence_threshold: The confidence threshold for the detections
        @type confidence_threshold: float
        @param num_classes: The number of classes in the model
        @type num_classes: int
        @param iou_threshold: The intersection over union threshold
        @type iou_threshold: float
        """
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

    def setConfidenceThreshold(self, threshold):
        """Sets the confidence score threshold for detected faces.

        @param threshold: Confidence score threshold for detected faces.
        @type threshold: float
        """
        self.confidence_threshold = threshold

    def setNumClasses(self, num_classes):
        """Sets the number of classes in the model.

        @param numClasses: The number of classes in the model.
        @type numClasses: int
        """
        self.num_classes = num_classes

    def setIouThreshold(self, iou_threshold):
        """Sets the intersection over union threshold.

        @param iou_threshold: The intersection over union threshold.
        @type iou_threshold: float
        """
        self.iou_threshold = iou_threshold

    def run(self):
        while self.isRunning():
            try:
                nnDataIn : dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break # Pipeline was stopped, no more data
            # Get all the layer names
            layer_names = nnDataIn.getAllLayerNames()

            outputs_names = sorted([name for name in layer_names if "_yolo" in name])
            outputs_values = [nnDataIn.getTensor(o, dequantize=True).astype(np.float32) for o in outputs_names]

            if any("kpt_output" in name for name in layer_names):
                mode = KPTS_MODE
                # Get the keypoint outputs
                kpts_output_names = sorted([name for name in layer_names if "kpt_output" in name])
                kpts_outputs = [nnDataIn.getTensor(o, dequantize=True).astype(np.float32) for o in kpts_output_names]
            elif any("_masks" in name for name in layer_names):
                mode = SEG_MODE
                # Get the segmentation outputs
                mask_outputs = sorted([name for name in layer_names if "_masks" in name])
                masks_outputs_values = [nnDataIn.getTensor(o, dequantize=True).astype(np.float32) for o in mask_outputs]
                protos_output = nnDataIn.getTensor("protos_output", dequantize=True).astype(np.float32)
                protos_len = protos_output.shape[1]

            if len(outputs_values[0].shape) != 4:
                # RVC4
                outputs_values = [o.transpose((2, 0, 1))[np.newaxis, ...] for o in outputs_values]
                if mode == KPTS_MODE:
                    kpts_outputs = [o[np.newaxis, ...] for o in kpts_outputs]
                elif mode == SEG_MODE:
                    protos_output = protos_output.transpose((2, 0, 1))[np.newaxis, ...]
                    protos_len = protos_output.shape[1]
                    masks_outputs_values = [o.transpose((2, 0, 1))[np.newaxis, ...] for o in masks_outputs_values]

            print("Outputs: ", outputs_names, ", shapes: ", [v.shape for v in outputs_values])
            if mode == KPTS_MODE:
                print("kpts_outputs: ", [o.shape for o in kpts_outputs])
            elif mode == SEG_MODE:
                print("Masks outputs: ", outputs_names, ", shapes: ", [v.shape for v in masks_outputs_values])
                print("Protos output shape: ", protos_output.shape)
            
            # Decode the outputs
            results = decode_yolo_output(
                outputs_values, 
                [8, 16, 32], 
                [None, None, None], 
                kpts=kpts_outputs if mode == KPTS_MODE else None,
                conf_thres=self.confidence_threshold,
                iou_thres=self.iou_threshold,
                num_classes=self.num_classes
            )

            bboxes, labels, scores, additional_output = [], [], [], []
            for i in range(results.shape[0]):
                print("Results shape: ", results[i].shape)
                bbox, conf, label, other = results[i, :4].astype(int), results[i, 4], results[i, 5].astype(int), results[i, 6:]
                
                bboxes.append(bbox)
                labels.append(int(label))
                scores.append(conf)

                if mode == KPTS_MODE:
                    kpts = parse_kpts(other)
                    additional_output.append(kpts)
                elif mode == SEG_MODE:
                    seg_coeff = other.astype(int)
                    hi, ai, xi, yi = seg_coeff
                    mask_coeff = masks_outputs_values[hi][0, ai*protos_len:(ai+1)*protos_len, yi, xi]
                    print("Mask coeff shape: ", mask_coeff.shape, mask_coeff[:5], type(mask_coeff[0]))
                    mask = process_single_mask(protos_output[0], mask_coeff, 0.5)
                    print("Mask shape: ", mask.shape)
                    additional_output.append(mask)

            if mode == KPTS_MODE:
                detections_message = create_detection_message(
                    np.array(bboxes),
                    np.array(scores),
                    labels,
                    keypoints=additional_output
                )
            elif mode == SEG_MODE:
                detections_message = create_detection_message(
                    np.array(bboxes),
                    np.array(scores),
                    labels,
                    masks=additional_output
                )
            self.out.send(detections_message)

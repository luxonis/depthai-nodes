import depthai as dai
import numpy as np

from ..messages.creators import create_segmentation_message

class SegmentationParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

    def run(self):
        """
        Postprocessing logic for Segmentation model with `num_classes` classes including background at index 0.

        Returns:
            Segmenation mask with `num_classes` classes, 0 - background.
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            output_layer_names = output.getAllLayerNames()
            
            if len(output_layer_names) != 1:
                raise ValueError(f"Expected 1 output layer, got {len(output_layer_names)}.")
            
            segmentation_mask = output.getTensor(output_layer_names[0])[0]  # num_clases x H x W

            if len(segmentation_mask.shape) != 3:
                raise ValueError(f"Expected 3D output tensor, got {len(segmentation_mask.shape)}D.")
            
            if segmentation_mask.shape[0] == 1:
                segmentation_mask = np.vstack((np.zeros((1, segmentation_mask.shape[1], segmentation_mask.shape[2]), dtype=np.float32), segmentation_mask))

            class_map = np.argmax(segmentation_mask, axis=0).reshape(segmentation_mask.shape[1], segmentation_mask.shape[2], 1).astype(np.uint8)

            imgFrame = create_segmentation_message(class_map)
            self.out.send(imgFrame)
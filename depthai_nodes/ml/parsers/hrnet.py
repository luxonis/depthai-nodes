import depthai as dai
import numpy as np
import cv2

from ..messages.creators import create_keypoints_message


class HRNetParser(dai.node.ThreadedHostNode):
    def __init__(self, score_threshold=0.5, input_size=[256, 256], heatmap_size=[64, 64]):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.score_threshold = score_threshold

    def setScoreThreshold(self, threshold):
        self.score_threshold = threshold

    def run(self):
        """Postprocessing logic for HRNet pose estimation model. The code is inspired by https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation
        
        Returns:
            ...
        """

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            img_width, img_height = self.input_size

            heatmaps = output.getTensor("heatmaps", dequantize=True)
            
            if len(heatmaps.shape) == 4: # add new axis for batch size
                heatmaps = heatmaps[0]

            if heatmaps.shape[2] == 16: # HW_ instead of _HW
                heatmaps = heatmaps.transpose(2, 0, 1)

            _, map_h, map_w = heatmaps.shape

            # Find the maximum value in each of the heatmaps and its location
            max_vals = np.array([np.max(heatmap) for heatmap in heatmaps])
            keypoints = np.array([np.unravel_index(heatmap.argmax(), heatmap.shape)
                                for heatmap in heatmaps])
            keypoints = keypoints.astype(np.float32)
            keypoints[max_vals < self.score_threshold] = np.array([np.nan, np.nan])

            # Scale keypoints to the image size
            # TODO: remove and have relative keypoint values? e.g. * np.array([64 / map_w, 64 / map_h]) to get relative values?
            keypoints = keypoints[:, ::-1] * np.array([img_width / map_w, img_height / map_h])

            keypoints_msg = create_keypoints_message(
                keypoints=keypoints,
                #scores=max_vals, # TODO: add scores
                #confidence_threshold=self.confidence_threshold # TODO: add confidence threshold
            )

            self.out.send(keypoints_msg)

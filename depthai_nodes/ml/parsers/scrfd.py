import cv2
import depthai as dai
import numpy as np

from ..messages.creators import create_detection_message


class SCRFDParser(dai.node.ThreadedHostNode):
    def __init__(self, score_threshold=0.5, nms_threshold=0.5, top_k=100):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def setConfidenceThreshold(self, threshold):
        self.score_threshold = threshold

    def setNMSThreshold(self, threshold):
        self.nms_threshold = threshold

    def setTopK(self, top_k):
        self.top_k = top_k

    def run(self):
        """Postprocessing logic for SCRFD model.

        Returns:
            ...
        """

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            print("SCRFD node")
            print(f"Layer names = {output.getAllLayerNames()}")

            score_8 = output.getTensor("score_8").flatten().astype(np.float32)
            score_16 = output.getTensor("score_16").flatten().astype(np.float32)
            score_32 = output.getTensor("score_32").flatten().astype(np.float32)
            bbox_8 = (
                output.getTensor("bbox_8").reshape(len(score_8), 4).astype(np.float32)
            )
            bbox_16 = (
                output.getTensor("bbox_16").reshape(len(score_16), 4).astype(np.float32)
            )
            bbox_32 = (
                output.getTensor("bbox_32").reshape(len(score_32), 4).astype(np.float32)
            )
            kps_8 = (
                output.getTensor("kps_8").reshape(len(score_8), 5, 2).astype(np.float32)
            )
            kps_16 = (
                output.getTensor("kps_16")
                .reshape(len(score_16), 5, 2)
                .astype(np.float32)
            )
            kps_32 = (
                output.getTensor("kps_32")
                .reshape(len(score_32), 5, 2)
                .astype(np.float32)
            )

            bboxes = []
            keypoints = []

            for i in range(len(score_8)):
                y = int(np.floor(i / 80)) * 4
                x = (i % 160) * 4
                bbox = bbox_8[i]
                xmin = int(x - bbox[0] * 8)
                ymin = int(y - bbox[1] * 8)
                xmax = int(x + bbox[2] * 8)
                ymax = int(y + bbox[3] * 8)
                kps = kps_8[i]
                kps_batch = []
                for kp in kps:
                    kpx = int(x + kp[0] * 8)
                    kpy = int(y + kp[1] * 8)
                    kps_batch.append([kpx, kpy])
                keypoints.append(kps_batch)
                bbox = [xmin, ymin, xmax, ymax]
                bboxes.append(bbox)

            for i in range(len(score_16)):
                y = int(np.floor(i / 40)) * 8
                x = (i % 80) * 8
                bbox = bbox_16[i]
                xmin = int(x - bbox[0] * 16)
                ymin = int(y - bbox[1] * 16)
                xmax = int(x + bbox[2] * 16)
                ymax = int(y + bbox[3] * 16)
                kps = kps_16[i]
                kps_batch = []
                for kp in kps:
                    kpx = int(x + kp[0] * 16)
                    kpy = int(y + kp[1] * 16)
                    kps_batch.append([kpx, kpy])
                keypoints.append(kps_batch)
                bbox = [xmin, ymin, xmax, ymax]
                bboxes.append(bbox)

            for i in range(len(score_32)):
                y = int(np.floor(i / 20)) * 16
                x = (i % 40) * 16
                bbox = bbox_32[i]
                xmin = int(x - bbox[0] * 32)
                ymin = int(y - bbox[1] * 32)
                xmax = int(x + bbox[2] * 32)
                ymax = int(y + bbox[3] * 32)
                kps = kps_32[i]
                kps_batch = []
                for kp in kps:
                    kpx = int(x + kp[0] * 32)
                    kpy = int(y + kp[1] * 32)
                    kps_batch.append([kpx, kpy])
                keypoints.append(kps_batch)
                bbox = [xmin, ymin, xmax, ymax]
                bboxes.append(bbox)

            scores = np.concatenate([score_8, score_16, score_32])
            indices = cv2.dnn.NMSBoxes(
                bboxes,
                list(scores),
                self.score_threshold,
                self.nms_threshold,
                top_k=self.top_k,
            )
            bboxes = np.array(bboxes)[indices]
            keypoints = np.array(keypoints)[indices]
            scores = scores[indices]

            detection_msg = create_detection_message(bboxes, scores, None, None)
            self.out.send(detection_msg)

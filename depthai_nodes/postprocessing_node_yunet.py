import depthai as dai
import numpy as np
import cv2

from ..custom_messages.img_detections import ImgDetectionsWithKeypoints

class YuNetParser(dai.node.ThreadedHostNode):
    def __init__(
        self,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        input_size=(640, 640), # WH
        strides=[8, 16, 32],
    ):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)

        self.input_size = input_size
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def setConfidenceThreshold(self, threshold):
        self.score_threshold = threshold

    def setNMSThreshold(self, threshold):
        self.nms_threshold = threshold

    def setTopK(self, top_k):
        self.top_k = top_k

    def setInputSize(self, width, height):
        self.input_size = (width, height)

    def setStrides(self, strides):
        self.strides = strides

    def run(self):
        """
        Postprocessing logic for YuNet model.

        Returns:
            ...
        """

        while self.isRunning():

            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException as e:
                break  # Pipeline was stopped

            detections = []
            for stride in self.strides:
                cols = int(self.input_size[1] / stride)  # w/stride
                rows = int(self.input_size[0] / stride)  # h/stride

                # Extract output blobs
                cls = output.getTensor(f"cls_{stride}").flatten()
                obj = output.getTensor(f"obj_{stride}").flatten()
                bbox = output.getTensor(f"bbox_{stride}").squeeze(0)
                kps = output.getTensor(f"kps_{stride}").squeeze(0)

                # Iterate over each grid cell
                for r in range(rows):
                    for c in range(cols):
                        idx = r * cols + c

                        # Decode scores
                        cls_score = np.clip(cls[idx], 0, 1)
                        obj_score = np.clip(obj[idx], 0, 1)
                        score = np.sqrt(cls_score * obj_score)

                        # Decode bounding box
                        cx = (c + bbox[idx, 0]) * stride
                        cy = (r + bbox[idx, 1]) * stride
                        w = np.exp(bbox[idx, 2]) * stride
                        h = np.exp(bbox[idx, 3]) * stride
                        x1 = cx - w / 2
                        y1 = cy - h / 2

                        # Decode landmarks
                        landmarks = []
                        for n in range(5):  # loop 5 times for 5 keypoints
                            lx = (kps[idx, 2 * n] + c) * stride
                            ly = (kps[idx, 2 * n + 1] + r) * stride
                            landmarks.extend([lx, ly])

                        # Append detection result including landmarks
                        detection = [x1, y1, w, h] + landmarks + [score]
                        if score > self.score_threshold:
                            detections.append(detection)

            # Convert results to numpy array
            detections = np.array(detections, dtype=np.float32)

            # Perform non-maximum suppression
            if len(detections) > 1:
                detection_boxes = detections[:, :4]
                detection_scores = detections[:, -1]
                indices = cv2.dnn.NMSBoxes(
                    list(detection_boxes),
                    list(detection_scores),
                    self.score_threshold,
                    self.nms_threshold,
                    top_k=self.top_k,
                )
                detections = detections[indices].reshape(-1, 15)

            img_detection_list = []
            landmarks_list = []
            for detection in detections:
                img_detection = dai.ImgDetection()
                img_detection.label=0
                img_detection.confidence=detection[-1]
                img_detection.xmin=detection[0]
                img_detection.ymin=detection[1]
                img_detection.xmax=detection[0] + detection[2]
                img_detection.ymax=detection[1] + detection[3]
                img_detection_list.append(img_detection)
                landmarks_list.append(detection[4:15])

            #detectionMessage = dai.ImgDetections()
            detectionMessage = ImgDetectionsWithKeypoints()
            detectionMessage.detections = img_detection_list
            detectionMessage.keypoints = landmarks_list

            self.out.send(detectionMessage)

import math

import cv2
import depthai as dai
import numpy as np

class ClassificationParser(dai.node.ThreadedHostNode):
    def __init__(self, labels: list = [], threshold: float = 0.5, top_k: int = 1):
        dai.node.ThreadedHostNode.__init__(self)
        self.input = dai.Node.Input(self)
        self.out = dai.Node.Output(self)
        self.threshold = threshold
        self.labels = np.array(labels)
        self.top_k = top_k
        self.nr_classes = len(labels)

    def setLabels(self, labels):
        self.labels = labels
        self.nr_classes = len(labels)
    
    def setThreshold(self, threshold):
        self.threshold = threshold
    
    def setTopK(self, top_k):
        self.top_k = top_k


    def run(self):
        """ Postprocessing logic for Classification model.

        Parameters
        ----------
        labels : list
            List of class labels.
        threshold : float
            Minimum confidence threshold for a class to be considered valid.
        top_k : int
            Number of top classes to return.
        
        Returns
        -------
            result: ndarray
                2D array containing top k classes and (optionally) their scores.

        """

        while self.isRunning():
            try:
                output: dai.NNData = self.input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped    
            
            output_layer_names = output.getAllLayerNames()
            if len(output_layer_names) != 1:
                raise ValueError(f"Expected 1 output layer, got {len(output_layer_names)}.")
            
            scores = output.getTensor(output_layer_names[0])
            scores = np.array(scores).flatten()

            if len(scores) != self.nr_classes:
                raise ValueError(f"Expected {self.nr_classes} scores, got {len(scores)}.")
            
            scores = scores[scores >= self.threshold]

            top_k_args = np.argsort(scores)[::-1][:self.top_k]
            top_k_scores = scores[top_k_args]

            classes = np.expand_dims(top_k_scores, axis=1)
            if self.labels:
                top_k_labels = self.labels[top_k_args]
                classes = np.vstack((top_k_labels, top_k_scores)).T

            # make message
            

            


import math

import cv2
import depthai as dai
import numpy as np

# from ..messages.creators import create_classification_message

class ClassificationMessage(dai.ADatatype):
    def __init__(self):
        dai.ADatatype.__init__(self)
        self.labels = []
        self.scores = []
        self.combined_results = []
    
    def setLabels(self, labels):
        self.labels = np.array(labels, dtype= np.str_)
        
    def setScores(self, scores):
        self.scores = scores
    
    def setCombinedResults(self, combined_results):
        self.combined_results = combined_results
        
    def getLabels(self):
        return self.labels
    
    def getScores(self):
        return self.scores
    
    def getCombinedResults(self):
        return self.combined_results
    

def create_classification_message(scores: np.array, labels: np.array = []) -> dai.ADatatype:
    # msg = dai.Buffer()
    
    # combined_results = list(scores)
    # if len(labels) == len(scores):
    #     combined_results = [[labels[i], scores[i]] for i in range(len(labels))]
    # print(combined_results)
    # msg.setData(combined_results)
    msg = ClassificationMessage()
    msg.setLabels(labels)
    msg.setScores(scores)
    if len(labels) == len(scores):
        combined_results = [[labels[i], scores[i]] for i in range(len(labels))]
        msg.setCombinedResults(combined_results)
    
    return msg

class ClassificationParser(dai.node.ThreadedHostNode):
    def __init__(self, class_labels: list = [], top_k: int = 1, threshold: float = 0):
        dai.node.ThreadedHostNode.__init__(self)
        self.out = self.createOutput()
        self.input = self.createInput()
        # self.input = dai.Node.Input(self)
        # self.out = dai.Node.Output(self)
        self.threshold = threshold
        self.class_labels = np.array(class_labels)
        self.top_k = top_k
        self.nr_classes = len(class_labels)
        
        self.checkTypes()
        
    def checkTypes(self):
        if self.top_k > self.nr_classes and self.nr_classes != 0:
            raise ValueError(f"Top k ({self.top_k}) is greater than number of classes ({self.nr_classes}).")
        
        if self.threshold < 0 or self.threshold >= 1:
            raise ValueError(f"Threshold should be between 0 and 1, got {self.threshold}.")
        
        if self.top_k <= 0:
            raise ValueError(f"Top k should be a positive integer, got {self.top_k}.")

    def setLabels(self, class_labels):
        self.class_labels = class_labels
        self.nr_classes = len(class_labels)
        self.checkTypes()
        
    def setThreshold(self, threshold):
        self.threshold = threshold
        self.checkTypes()
    
    def setTopK(self, top_k):
        self.top_k = top_k
        self.checkTypes()


    def run(self):
        """ Postprocessing logic for Classification model.

        Parameters
        ----------
        top_k : int
            Number of classes to return.
        class_labels : list
            List of class labels.
        threshold : float
            Minimum confidence threshold for a class to be considered valid.
            Not used by default.
        
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

            if len(scores) != self.nr_classes and self.nr_classes != 0:
                raise ValueError(f"Number of labels and scores mismatch. Provided {self.nr_classes} labels and {len(scores)} scores.")
            
            
            top_k_args = np.argsort(scores)[::-1][:self.top_k]
            top_k_scores = scores[top_k_args]
            
            top_k_scores = top_k_scores[top_k_scores >= self.threshold]
            top_k_args = top_k_args[top_k_scores >= self.threshold]
            
            
            # if len(top_k_scores) < self.top_k:
            #     raise ValueError(f"No scores meet criteria, list is empty.")
            
            top_k_labels = []
            if len(self.class_labels) > 0:
                top_k_labels = self.class_labels[top_k_args]
            

            msg = create_classification_message(top_k_scores, top_k_labels)
            
            self.out.send(msg)
            

            


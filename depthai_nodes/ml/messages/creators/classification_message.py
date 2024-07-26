import depthai as dai
import numpy as np

def create_classification_message(scores: np.array, labels: np.array = []) -> dai.ADatatype:
    msg = dai.ADatatype()
    
    msg.labels = labels
    msg.scores = scores
    msg.combined_results = []
    if len(labels) == len(scores):
        msg.combined_results = [[labels[i], scores[i]] for i in range(len(labels))]
    
    return msg

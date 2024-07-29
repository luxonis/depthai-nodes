import depthai as dai
import numpy as np

from ...messages import ClassificationMessage


def create_classification_message(scores, classes) -> dai.Buffer:
    msg = ClassificationMessage()

    sorted_args = np.argsort(scores)[::-1]
    scores = scores[sorted_args]
    classes = classes[sorted_args]

    msg.sortedClasses = [
        [str(classes[i]), float(scores[i])] for i in range(len(classes))
    ]

    return msg

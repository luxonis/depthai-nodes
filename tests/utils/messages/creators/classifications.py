from typing import List

import depthai_nodes.message.creators as creators

from .constants import CLASSIFICATION


def create_classifications(
    classes: List[str] = CLASSIFICATION["classes"],
    scores: List[float] = CLASSIFICATION["scores"],
):
    return creators.create_classification_message(classes=classes, scores=scores)


def create_classifications_sequence(
    classes: List[str] = CLASSIFICATION["classes"],
    scores: List[float] = [
        CLASSIFICATION["scores"],
    ]
    * CLASSIFICATION["sequence_num"],
):
    return creators.create_classification_sequence_message(
        classes=classes, scores=scores
    )

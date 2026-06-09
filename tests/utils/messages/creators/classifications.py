import depthai_nodes.message.creators as creators

from .constants import CLASSIFICATION


def create_classifications(
    classes: list[str] = CLASSIFICATION["classes"],
    scores: list[float] = CLASSIFICATION["scores"],
):
    return creators.create_classification_message(classes=classes, scores=scores)


def create_classifications_sequence(
    classes: list[str] = CLASSIFICATION["classes"],
    scores: list[float] = [
        CLASSIFICATION["scores"],
    ]
    * CLASSIFICATION["sequence_num"],
):
    return creators.create_classification_sequence_message(
        classes=classes, scores=scores
    )

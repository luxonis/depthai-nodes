import depthai_nodes.message.creators as creators

from .constants import SCORES


def create_regression(predictions: list[float] = SCORES):
    return creators.create_regression_message(predictions=predictions)

from typing import List

import depthai_nodes.message.creators as creators

from .constants import SCORES


def create_regression(predictions: List[float] = SCORES):
    return creators.create_regression_message(predictions=predictions)

from typing import Optional

import numpy as np


def softmax(
    x: np.ndarray, axis: Optional[int] = None, keep_dims: bool = False
) -> np.ndarray:
    """Compute the softmax of an array. The softmax function is defined as: softmax(x) =
    exp(x) / sum(exp(x))

    @param x: The input array.
    @type x: np.ndarray
    @param axis: Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is negative it counts
        from the last to the first axis.
    @type axis: int
    @param keep_dims: If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will broadcast
        correctly against the input array.
    @type keep_dims: bool
    @return: The softmax of the input array.
    @rtype: np.ndarray
    """
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=keep_dims)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function.

    @param x: Input tensor.
    @type x: np.ndarray
    @return: A result tensor after applying a sigmoid function on the given input.
    @rtype: np.ndarray
    """
    return 1 / (1 + np.exp(-x))

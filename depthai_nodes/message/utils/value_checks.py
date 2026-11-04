"""Type and value checks shared by the box measurement messages and nodes.

A float accepts any ``numbers.Real`` and an int any ``numbers.Integral``, numpy
scalars included. Both reject ``bool`` and ``str``. Accepted values are returned
as built-in ``float`` or ``int``.
"""

import math
import numbers
from typing import Any


def check_float(name: str, value: Any) -> float:
    """Return a finite real number as a built-in ``float``.

    Args:
        name: Field name used in error messages.
        value: Value to check.

    Returns:
        The value as a ``float``.

    Raises:
        TypeError: If the value is a ``bool`` or not a ``numbers.Real``.
        ValueError: If the value is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def check_int(name: str, value: Any) -> int:
    """Return an integral number as a built-in ``int``.

    Args:
        name: Field name used in error messages.
        value: Value to check.

    Returns:
        The value as an ``int``.

    Raises:
        TypeError: If the value is a ``bool`` or not a ``numbers.Integral``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def check_instance(name: str, value: Any, expected: type) -> None:
    """Check that a value is an instance of the expected type.

    Args:
        name: Field name used in error messages.
        value: Value to check.
        expected: Required type.

    Raises:
        TypeError: If the value is not an instance of ``expected``.
    """
    if not isinstance(value, expected):
        raise TypeError(
            f"{name} must be {expected.__name__}, got {type(value).__name__}"
        )

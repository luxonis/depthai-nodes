"""The result message of ``AutoSupportPlane``."""

from __future__ import annotations

import depthai as dai

from .support_plane import SupportPlane


class AutoSupportPlaneResult(dai.Buffer):
    """One finished cycle of ``AutoSupportPlane``: a verdict and the plane in use.

    It is a Python ``Buffer`` subclass, so it travels between host nodes only;
    over XLink or a ``RemoteConnection`` it arrives as a bare ``Buffer``.

    Args:
        passed: Whether the cycle found or confirmed a support plane.
        plane: The support plane in use, or ``None`` when there is none.

    Raises:
        TypeError: If ``passed`` is not a ``bool`` or ``plane`` is neither a
            `SupportPlane` nor ``None``.
        ValueError: If ``passed`` is ``True`` with no plane.
    """

    def __init__(self, passed: bool, plane: SupportPlane | None) -> None:
        """Create a result and check that its fields are consistent.

        Args:
            passed: Whether the cycle found or confirmed a support plane.
            plane: The support plane in use, or ``None``.

        Raises:
            TypeError: If ``passed`` is not a ``bool`` or ``plane`` is neither
                a `SupportPlane` nor ``None``.
            ValueError: If ``passed`` is ``True`` with no plane.
        """
        super().__init__()
        if not isinstance(passed, bool):
            raise TypeError(f"passed must be a bool, got {type(passed).__name__}")
        if plane is not None and not isinstance(plane, SupportPlane):
            raise TypeError(
                f"plane must be SupportPlane or None, got {type(plane).__name__}"
            )
        if passed and plane is None:
            raise ValueError("A passed result must have a plane")
        self._passed = passed
        self._plane = plane

    @property
    def passed(self) -> bool:
        """Whether the cycle found or confirmed a support plane."""
        return self._passed

    @property
    def plane(self) -> SupportPlane | None:
        """The support plane in use, or ``None`` when there is none."""
        return self._plane

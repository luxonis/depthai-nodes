"""Check for open3d, which the box measurement nodes need."""

INSTALL_HINT = "pip install depthai-nodes[box-measurement]"


def require_open3d(node: str) -> None:
    """Check that open3d is installed.

    Args:
        node: Name of the node that needs open3d, used in the error message.

    Raises:
        ImportError: If open3d is not installed.
    """
    try:
        import open3d  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"{node} requires open3d. Install it with `{INSTALL_HINT}`."
        ) from e

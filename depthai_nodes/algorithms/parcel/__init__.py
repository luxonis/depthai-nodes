"""The single bundled parcel algorithm and its private frozen configuration.

There is no runtime algorithm selection or settings override. The optional parcel extra
is loaded only when the computation callbacks are created.
"""

import json
from pathlib import Path

from .implementation.components import (
    Components,
)
from .implementation.scene import (
    CameraRole,
    CameraView,
    CandidateDecision,
    CandidateResult,
    Cuboid,
    Matrix3,
    Matrix4,
    PointCloud,
    Scene,
    SceneError,
    Segment,
    SupportPlane,
    Vec3,
    project_points,
)

ALGORITHM_ID = "algorithm-5e4abc3a09b54148950317cdeac60ba2"
IMPLEMENTATION_NAME = "implementation"
_ALGORITHM_NAME = "fast_baseline_oak4_visibility_01"
_CONFIGURATION_VERSION = "v002"


def create_components() -> Components:
    """Create callbacks bound to the bundled settings.

    Returns:
        A fresh component instance. Complete a scene call before reusing it.

    Raises:
        ImportError: If the optional parcel computation dependencies are absent.
    """
    try:
        from .implementation import (
            create_components as create,
        )

        settings_path = (
            Path(__file__).parent / IMPLEMENTATION_NAME / "scoped-settings.json"
        )
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        return create(settings["configuration"])
    except ModuleNotFoundError as exc:
        if exc.name == "scipy" or (exc.name or "").startswith("scipy."):
            raise ImportError(
                'Parcel nodes require scipy. Install "depthai-nodes[parcel]".'
            ) from exc
        raise


def validate_manifest() -> dict:
    """Validate bundled provenance, hashes, configuration, and node bindings.

    Returns:
        The validated manifest.

    Raises:
        ValueError: If the package differs from its declared provenance.
    """
    from .manifest import validate_manifest as validate

    return validate()


__all__ = [
    "ALGORITHM_ID",
    "IMPLEMENTATION_NAME",
    "Components",
    "CameraRole",
    "CameraView",
    "CandidateDecision",
    "CandidateResult",
    "Cuboid",
    "Matrix3",
    "Matrix4",
    "PointCloud",
    "Scene",
    "SceneError",
    "Segment",
    "SupportPlane",
    "Vec3",
    "create_components",
    "project_points",
    "validate_manifest",
]

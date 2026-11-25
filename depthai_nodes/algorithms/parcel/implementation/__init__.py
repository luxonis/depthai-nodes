"""Frozen OAK-4 visibility computation, mechanically ported from Luxonis BoxLabRunners.

The factory is imported only when used so the parcel extra remains optional. See
ATTRIBUTION.md for source identity and adaptations.
"""


def create_components(configuration):
    """Bind the frozen factory to an internal configuration."""
    from .factory import create_components as create

    return create(configuration)

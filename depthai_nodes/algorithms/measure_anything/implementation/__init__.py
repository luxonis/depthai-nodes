"""Frozen OAK-4 visibility computation, mechanically ported from Luxonis.

The factory is imported only when used so the measure-anything extra remains optional.
The package manifest records source identity and port adaptations.
"""


def create_components(configuration):
    """Bind the frozen factory to an internal configuration."""
    from .factory import create_components as create

    return create(configuration)

from importlib.metadata import PackageNotFoundError, version

from .core import serialize_fn_map

__all__ = ["serialize_fn_map"]

try:
    __version__ = version("brainsets")
except PackageNotFoundError:  # pragma: no cover
    # This can happen if someone is importing brainsets without installing
    __version__ = "unknown"  # pragma: no cover

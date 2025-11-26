from importlib import metadata

try:
    __version__ = metadata.version("spec-driven-development-agent")
except metadata.PackageNotFoundError:  # pragma: no cover - dev installs only
    __version__ = "0.0.0"

__all__ = ["__version__"]



"""Shared locations for application data and isolated dependency caches."""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_data_root(value=None) -> Path:
    """Resolve an explicit root, the environment override, or the source root.

    Defaults never depend on the launching directory. Relative overrides are
    relative to the caller's working directory, as with ordinary CLI paths.
    Reading the environment on demand also supports embedded application use.
    """
    configured = value if value is not None else os.environ.get("AUTOREPRO_DATA_ROOT", "").strip()
    return Path(configured or PROJECT_ROOT / "data").expanduser().absolute()


def get_deps_root(value=None) -> Path:
    """An explicit cache overrides AUTOREPRO_DEPS_ROOT, then <data>/deps."""
    configured = value if value is not None else os.environ.get("AUTOREPRO_DEPS_ROOT", "").strip()
    return Path(configured or get_data_root() / "deps").expanduser().absolute()

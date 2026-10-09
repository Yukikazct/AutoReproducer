"""Identify the running interpreter even in stripped Windows environments."""
import platform
import sys
import sysconfig


def runtime_fingerprint():
    # Windows platform.machine() depends on environment variables which some
    # launchers omit. The interpreter's build platform also avoids confusing a
    # 32-bit Python with its 64-bit host when selecting native dependencies.
    build = sysconfig.get_platform().lower()
    if sys.platform == "win32":
        machine = {"win-amd64": "AMD64", "win32": "x86",
                   "win-arm64": "ARM64"}.get(build)
        if machine is None:
            raise RuntimeError(f"Unsupported Windows Python platform: {build}")
    else:
        machine = platform.machine() or build
    return f"{sys.implementation.cache_tag}-{sys.platform}-{machine}"

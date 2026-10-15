"""Identify the running interpreter even in stripped Windows environments."""
import platform
import json
import subprocess
import sys
import sysconfig


def runtime_fingerprint(executable=None):
    if executable and executable != sys.executable:
        result = subprocess.run([str(executable), '-I', '-c',
            'import json,platform,sys,sysconfig; print(json.dumps([sys.implementation.cache_tag,sys.platform,platform.machine() or sysconfig.get_platform(),sys.version_info[:2]]))'],
            capture_output=True, text=True, timeout=15, check=True)
        tag, system, machine, version = json.loads(result.stdout.strip())
        if tuple(version) not in {(3, 9), (3, 10), (3, 11), (3, 12)} or not tag.startswith('cpython-'):
            raise ValueError('Training Python must be CPython 3.9–3.12')
        return f'{tag}-{system}-{machine}'
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

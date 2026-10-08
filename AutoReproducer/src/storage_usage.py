"""有来源和统计边界的文件体积；不推测整个环境或本次新增磁盘占用。"""
import os
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def unmeasured_disk_usage():
    return {"status": "not_measured", "components": [],
            "note": "尚未读取实际文件或镜像大小，不提供模型估值。"}


def directory_usage(path, key, label, *, exclude=(), shared=False, retained=True,
                    max_entries=200000, timeout=5):
    """读取文件长度，不跟随符号链接；硬链接只计一次。失败不能伪装为零。"""
    root = Path(path)
    result = {"key": key, "label": label, "source": str(root), "basis": "file_stat",
              "status": "not_measured", "bytes": None, "allocated_bytes": None,
              "shared": shared, "retained": retained, "files": 0}
    try:
        root_stat = root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            result["note"] = "目录不存在或为符号链接，未统计其目标。"
            return result
    except OSError:
        result["note"] = "目录无法读取，未测量。"
        return result
    deadline = time.monotonic() + timeout
    pending, seen, errors = [root], set(), []
    logical, allocated, visited = 0, 0, 0
    blocks_available = True
    limited = False
    scanned = False
    while pending and not limited:
        if time.monotonic() > deadline:
            limited = True
            break
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                scanned = True
                for entry in entries:
                    visited += 1
                    if visited > max_entries or time.monotonic() > deadline:
                        limited = True
                        break
                    if directory == root and entry.name in exclude:
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                        # Windows DirEntry.stat() may omit the file ID; os.stat()
                        # obtains it so distinct hardlink names share one identity.
                        if not info.st_ino:
                            info = os.stat(entry.path, follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(info.st_mode):
                            identity = (info.st_dev, info.st_ino) if info.st_ino else entry.path
                            if identity in seen:
                                continue
                            seen.add(identity)
                            logical += info.st_size
                            result["files"] += 1
                            blocks = getattr(info, "st_blocks", None)
                            if blocks is None:
                                blocks_available = False
                            else:
                                allocated += blocks * 512
                    except OSError:
                        errors.append(entry.name)
        except OSError:
            errors.append(directory.name)
    if not scanned:
        result["note"] = "目录无法遍历，未测量。"
        return result
    result.update({"status": "partial" if limited or errors else "measured",
                   "bytes": logical, "allocated_bytes": allocated if blocks_available else None})
    if limited or errors:
        result["note"] = "仅统计已读取文件；遇到读取错误或统计预算上限，结果不完整。"
    return result


def docker_image_usage(docker_cmd, image):
    """只读本地 Docker 元数据，Size 包含共享层，不代表 VM 实际新增空间。"""
    result = {"key": "docker_image", "label": "Docker 镜像内容体积", "source": image,
              "basis": "docker_image_inspect", "status": "not_measured", "bytes": None,
              "shared": True, "retained": True,
              "note": "Docker 返回的镜像逻辑大小，包含共享层；不等于本次新增磁盘占用。"}
    try:
        inspected = subprocess.run(
            [docker_cmd, "image", "inspect", "--format", "{{.Id}} {{.Size}}", image],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
        parts = (inspected.stdout or "").strip().split()
        if inspected.returncode != 0 or len(parts) != 2 or not parts[0].startswith("sha256:"):
            return result
        size = int(parts[1])
        if size < 0:
            return result
        result.update({"status": "measured", "bytes": size, "image_id": parts[0]})
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def disk_usage_snapshot(components):
    states = [component["status"] for component in components]
    measured = any(state in {"measured", "partial"} for state in states)
    return {"status": ("measured" if states and all(s == "measured" for s in states)
                       else "partial" if measured else "not_measured"),
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "components": components,
            "note": "统计时点为本阶段执行结束、临时工作区清理之前；各项不相加为总占用。"}


def format_bytes(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return "未测量"
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024 or unit == "TiB":
            return f"{value} B" if unit == "B" else f"{number:.2f} {unit}"
        number /= 1024

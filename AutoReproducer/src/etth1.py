"""Pinned ETTh1 acquisition; validate before promoting a file into the cache."""
import csv
import hashlib
import math
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.experiment_profiles import DATA_REVISION, DATA_URL

GIT_BLOB_SHA = "a52c4925778c07c1ef1a2cf6fd01594919717d9e"
API_URL = ("https://api.github.com/repos/zhouhaoyi/ETDataset/contents/"
           f"ETT-small/ETTh1.csv?ref={DATA_REVISION}")
HEADER = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def validate_csv(path: Path) -> dict:
    content = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if blob != GIT_BLOB_SHA:
        raise ValueError("ETTh1 文件与固定官方版本不符（校验失败）")
    previous, count = None, 0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != HEADER:
            raise ValueError("ETTh1 列结构不符")
        for row in reader:
            if len(row) != len(HEADER) or not all(math.isfinite(float(v)) for v in row[1:]):
                raise ValueError("ETTh1 含缺失或非有限数值")
            stamp = datetime.fromisoformat(row[0])
            if previous is not None and stamp - previous != timedelta(hours=1):
                raise ValueError("ETTh1 时间顺序或间隔不符")
            previous, count = stamp, count + 1
    if count != 17420:
        raise ValueError("ETTh1 时间序列不完整")
    return {"sha256": hashlib.sha256(content).hexdigest(), "rows": count,
            "bytes": len(content), "data_kind": "real", "source_url": DATA_URL,
            "source_revision": DATA_REVISION, "dataset_name": "ETTh1"}


def fetch_etth1(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "ETTh1.csv"
    if target.is_file():
        try:
            return {**validate_csv(target), "path": str(target), "state": "cached",
                    "detail": "真实 ETTh1 缓存校验通过"}
        except (OSError, ValueError):
            pass  # Never use a corrupt/partial cache entry.
    errors = []
    for url in (DATA_URL, API_URL):
        fd, name = tempfile.mkstemp(prefix=".ETTh1-", suffix=".part", dir=directory)
        os.close(fd)
        temporary = Path(name)
        try:
            proc = subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--connect-timeout", "10", "--max-time", "180",
                 "--retry", "1", "--retry-max-time", "240",
                 "-H", "Accept: application/vnd.github.raw+json",
                 "--output", str(temporary), url],
                capture_output=True, text=True, timeout=425)
            if proc.returncode:
                raise ValueError((proc.stderr or "下载失败")[-300:])
            metadata = validate_csv(temporary)
            os.replace(temporary, target)
            return {**metadata, "path": str(target), "state": "downloaded",
                    "download_url": url, "detail": "官方 ETTh1 已下载并校验"}
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": "", "state": "unavailable", "data_kind": "real",
            "source_url": DATA_URL, "source_revision": DATA_REVISION,
            "detail": "ETTh1 下载失败: " + "; ".join(errors)}

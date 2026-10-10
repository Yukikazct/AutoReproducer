"""Bounded transport recovery and atomic publication for frozen preset inputs."""
import http.client
import math
from email.utils import parsedate_to_datetime
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def download_bytes(url, *, max_bytes, timeout_s, attempts=3):
    """Retry transport failures only; callers still verify fixed size/hash.

    Error bodies/reasons are omitted because proxy credentials can appear in
    nested transport errors. A long server Retry-After stops immediately rather
    than blocking preparation indefinitely or violating the publisher's limit.
    """
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("固定来源下载最多允许1至3次尝试")
    request = urllib.request.Request(url, headers={"User-Agent": "AutoReproducer"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return response.read(max_bytes + 1)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead) as error:
            status = error.code if isinstance(error, urllib.error.HTTPError) else None
            retry_after = None
            if isinstance(error, urllib.error.HTTPError):
                raw = error.headers.get("Retry-After") if error.headers is not None else None
                if isinstance(raw, str):
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        try:
                            retry_after = max(0, parsedate_to_datetime(raw).timestamp() - time.time())
                        except (TypeError, ValueError, OverflowError):
                            pass
                    if retry_after is not None and (not math.isfinite(retry_after) or retry_after < 0):
                        retry_after = None
                error.close()
            retryable = status is None or status in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == attempts or (retry_after is not None and retry_after > 5):
                detail = f"HTTP {status}" if status is not None else "网络连接或超时错误"
                raise RuntimeError(f"固定数据下载失败: {detail}；已尝试 {attempt} 次") from None
            time.sleep(max(2 ** (attempt - 1), retry_after or 0))


def atomic_cache_bytes(path, content):
    """Publish already verified bytes; a failure preserves the previous cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.part")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

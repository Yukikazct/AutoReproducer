"""Observe pip's owned log and download files without changing its process I/O."""
import os
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit
import uuid

from src.safety.paths import filesystem_path


def parse_pip_line(line):
    """Return only package facts, never URL credentials or arbitrary pip output."""
    if 'Installing collected packages:' in line:
        return {'phase': 'installing', 'package': None}
    request = re.search(r'"GET (\S+) HTTP/\S+"\s+200\s+(\d+)\s*$', line)
    download = re.search(r'Downloading (\S+)', line)
    if request is None and download is None:
        return None
    location = (request or download).group(1)
    try:
        filename = unquote(urlsplit(location).path.rsplit('/', 1)[-1])
        domain = urlsplit(location).hostname
        if domain is None and request is not None:
            host = re.search(r'https?://[^\s"\']+', line[:request.start()])
            domain = urlsplit(host.group()).hostname if host else None
    except ValueError:
        return None
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.+!-]*\.whl', filename):
        return None
    return {'phase': 'downloading', 'package': filename.split('-')[0],
            'file_name': filename, 'total_bytes': int(request.group(2)) if request else None,
            'source_domain': domain}


class DependencyInstallProgress:
    def __init__(self, work, callback, timeout_s, interval_s=1.0):
        self.work = Path(work)
        self.log_path = self.work / 'pip-progress.log'
        self.callback, self.timeout_s = callback, timeout_s
        self.interval_s = interval_s
        self.installation_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.last_emit = None
        self.last_activity_at = time.time()
        self.phase, self.package, self.file_name = 'resolving', None, None
        self.source_domain = None
        self._offset, self._pending = 0, ''
        self._totals = {}
        self._sample_file, self._sample_bytes, self._sample_at = None, None, None
        self.downloaded_bytes = self.download_speed_bps = None

    def _read_log(self):
        try:
            with Path(filesystem_path(self.log_path)).open('rb') as stream:
                size = os.fstat(stream.fileno()).st_size
                if size < self._offset:
                    self._offset, self._pending = 0, ''
                if size - self._offset > 128 * 1024:
                    self._offset, self._pending = size - 128 * 1024, ''
                stream.seek(self._offset)
                chunk = stream.read(128 * 1024)
                self._offset = stream.tell()
        except OSError:
            return
        if not chunk:
            return
        self.last_activity_at = time.time()
        lines = (self._pending + chunk.decode('utf-8', 'replace')).split('\n')
        self._pending = lines.pop()[-8192:]
        for line in lines:
            parsed = parse_pip_line(line)
            if parsed is None:
                continue
            self.phase, self.package = parsed['phase'], parsed['package']
            self.file_name = parsed.get('file_name')
            if parsed.get('source_domain'):
                self.source_domain = parsed['source_domain']
            if parsed.get('total_bytes') is not None:
                self._totals[self.file_name] = parsed['total_bytes']
                if len(self._totals) > 100:
                    self._totals.pop(next(iter(self._totals)))

    @staticmethod
    def _linked(path):
        info = path.lstat()
        return path.is_symlink() or bool(getattr(info, 'st_file_attributes', 0) & 0x400)

    def _download_size(self):
        if self.phase != 'downloading' or not self.file_name:
            return None
        # pip's unpack/download folders are shallow. Never traverse extracted
        # packages or an entire installed environment on every heartbeat.
        for directory in self.work.iterdir():
            if not directory.name.startswith(('pip-unpack-', 'pip-download-')) or self._linked(directory):
                continue
            candidate = Path(filesystem_path(directory / self.file_name))
            try:
                if not candidate.is_file() or self._linked(candidate):
                    continue
                with candidate.open('rb') as stream:
                    # Windows directory enumeration reports stale lengths while
                    # pip still holds its write handle; query the open file.
                    return os.fstat(stream.fileno()).st_size
            except OSError:
                continue
        return None

    def _sample(self, now):
        self._read_log()
        size = self._download_size()
        self.download_speed_bps = None
        if size is not None and self.file_name == self._sample_file and self._sample_bytes is not None:
            duration = now - self._sample_at
            self.download_speed_bps = max(0, size - self._sample_bytes) / duration if duration > 0 else None
            if size != self._sample_bytes:
                self.last_activity_at = time.time()
        elif size is not None:
            self.last_activity_at = time.time()
        self.downloaded_bytes = size
        self._sample_file, self._sample_bytes, self._sample_at = self.file_name, size, now

    def _emit(self, status, now):
        messages = {'resolving': '正在解析依赖或等待软件源响应',
                    'downloading': '正在下载依赖包', 'installing': '正在安装依赖并复制文件'}
        message = messages[self.phase] if status == 'running' else {
            'success': '本轮依赖命令已完成', 'error': '本轮依赖命令未成功',
            'cancelled': '本轮依赖命令已中断'}[status]
        event = {'type': 'dependency_install_progress', 'installation_id': self.installation_id,
                 'status': status, 'phase': self.phase, 'package': self.package,
                 'downloaded_bytes': self.downloaded_bytes,
                 'total_bytes': self._totals.get(self.file_name),
                 'download_speed_bps': self.download_speed_bps,
                 'elapsed_s': round(now - self.started, 3), 'timeout_s': self.timeout_s,
                 'last_activity_at': self.last_activity_at, 'message': message,
                 'source_domain': self.source_domain}
        self.callback(event)

    def __call__(self, process):
        now = time.monotonic()
        if self.last_emit is not None and now - self.last_emit < self.interval_s:
            return
        self.last_emit = now
        try:
            self._sample(now)
            self._emit('running', now)
        except Exception:
            # An observer must not kill a healthy download or leak its children.
            pass

    def finish(self, status):
        now = time.monotonic()
        try:
            self._sample(now)
        except Exception:
            pass
        try:
            self._emit(status, now)
        except Exception:
            pass

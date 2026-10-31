"""Observe author append-only text outputs without altering tracked examples.

Only direct calls from frozen author Python sources are eligible. The observer
preserves the original prefix and redirects subsequent accesses to a run-local
copy. Receipts expose only newly appended bytes as this execution's output.
"""
import builtins
import hashlib
import os
from pathlib import Path
import sys
import uuid


_LIMIT = 2 * 1024 * 1024
_SUFFIXES = {'.txt', '.log', '.csv', '.jsonl'}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _io_path(path):
    """Keep logical receipt paths portable; use extended paths only for I/O."""
    value = os.path.abspath(os.fspath(path))
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return Path(value)


class AppendOutputCapture:
    def __init__(self, workspace, runtime, sources, step_id, input_files=()):
        self.workspace = Path(workspace).resolve()
        self.runtime = Path(runtime).resolve()
        if not self.runtime.is_relative_to(self.workspace) or not step_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Output capture requires a run-local directory and a valid step ID')
        self.sources = sources
        self.step_id = step_id
        self.inputs = set(input_files)
        self.original_open = builtins.open
        self.files = {}
        self.handles = []
        self.attempt_id = uuid.uuid4().hex

    def install(self):
        builtins.open = self.open

    def close(self):
        builtins.open = self.original_open

    def open(self, file, mode='r', *args, **kwargs):
        if not isinstance(file, (str, bytes, os.PathLike)) or kwargs.get('opener') is not None:
            return self.original_open(file, mode, *args, **kwargs)
        try:
            path = Path(os.fsdecode(file)).absolute()
            resolved = path.resolve()
            relative = resolved.relative_to(self.workspace).as_posix()
            if (resolved.suffix.lower() not in _SUFFIXES or '.git' in resolved.parts
                    or relative in self.inputs or resolved.is_relative_to(self.runtime)):
                return self.original_open(file, mode, *args, **kwargs)
            if any(_io_path(parent).is_symlink() or (_io_path(parent).exists()
                       and getattr(_io_path(parent).lstat(), 'st_file_attributes', 0) & 0x400)
                   for parent in (path, *path.parents) if parent != self.workspace):
                return self.original_open(file, mode, *args, **kwargs)
            if relative in self.files:
                return self._redirect(self.files[relative], mode, args, kwargs)
            if not isinstance(mode, str) or 'a' not in mode or len(self.files) >= 16:
                return self.original_open(file, mode, *args, **kwargs)
            frame = sys._getframe(1)
            caller = Path(frame.f_code.co_filename).resolve()
            source = caller.relative_to(self.workspace).as_posix()
            if self.sources.get(source) != _sha(_io_path(caller).read_bytes()):
                return self.original_open(file, mode, *args, **kwargs)
            physical = _io_path(path)
            if physical.exists() and (not physical.is_file() or physical.stat().st_size > _LIMIT):
                return self.original_open(file, mode, *args, **kwargs)
            prefix = physical.read_bytes() if physical.exists() else b''
            # A tracked text file is preserved byte-for-byte; source changes
            # remain visible to the regular frozen-input verification.
            if relative in self.sources and self.sources[relative] != _sha(prefix):
                return self.original_open(file, mode, *args, **kwargs)
            directory = self.runtime / 'output_logs' / self.step_id / self.attempt_id
            _io_path(directory).mkdir(parents=True, exist_ok=True)
            mirror = directory / (_sha(relative.encode())[:24] + path.suffix)
            with self.original_open(_io_path(mirror), 'xb') as stream:
                stream.write(prefix)
            record = {'path': relative, 'source_path': source, 'source_sha256': self.sources[source],
                'source_line': frame.f_lineno, 'step_id': self.step_id,
                'captured_path': mirror.relative_to(self.workspace).as_posix(),
                'redirected': relative in self.sources,
                'original_bytes': len(prefix), 'original_sha256': _sha(prefix)}
            self.files[relative] = record
            return self._redirect(record, mode, args, kwargs)
        except (OSError, ValueError, TypeError):
            # Observation cannot suppress the author's normal file error.
            return self.original_open(file, mode, *args, **kwargs)

    def _redirect(self, record, mode, args, kwargs):
        handle = self.original_open(_io_path(self.workspace / (record['captured_path'] if record['redirected'] else record['path'])),
                                    mode, *args, **kwargs)
        self.handles.append(handle)
        return handle

    def finish(self):
        for handle in self.handles:
            if not handle.closed:
                handle.flush()
        records = []
        for record in self.files.values():
            path = _io_path(self.workspace / record['captured_path'])
            if not record['redirected']:
                original = _io_path(self.workspace / record['path'])
                if not original.is_file() or original.stat().st_size > _LIMIT:
                    continue
                with self.original_open(path, 'wb') as stream:
                    stream.write(original.read_bytes())
            if path.stat().st_size > _LIMIT:
                continue
            raw = path.read_bytes()
            offset = record['original_bytes']
            if len(raw) < offset or _sha(raw[:offset]) != record['original_sha256']:
                continue  # A rewritten log cannot be confused with new append output.
            suffix = raw[offset:]
            if not suffix:
                continue
            records.append({**record, 'captured_sha256': _sha(raw), 'captured_bytes': len(raw),
                'appended_sha256': _sha(suffix), 'appended_bytes': len(suffix)})
        return records

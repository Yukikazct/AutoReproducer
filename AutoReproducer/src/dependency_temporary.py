"""Task-owned pip scratch space; never sweep the system temporary directory."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

from filelock import FileLock, Timeout

from src.dependency_cache import cache_guard
from src.safety.files import remove_tree
from src.safety.paths import filesystem_path, workspace_path


TEMP_NAMESPACE = '.pip-temporary'
OWNER_MARKER = '.autorepro-pip-owner.json'
_NAME = re.compile(r'install-([a-f0-9]{32})')
ORPHAN_GRACE_SECONDS = 86400


def temporary_bytes(directory):
    """Count regular files without following pip-created links or junctions."""
    total, pending = 0, [Path(filesystem_path(directory))]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0)
                        & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
    return total


@contextmanager
def pip_temporary(root, *, on_cleanup=None):
    """Hold ownership until the installer and its descendants have returned."""
    root = workspace_path(root, '.', 'dependency cache root')
    with cache_guard(root):
        parent = workspace_path(root, TEMP_NAMESPACE, 'pip temporary namespace')
        parent.mkdir(exist_ok=True)
        token = uuid.uuid4().hex
        work = workspace_path(root, f'{TEMP_NAMESPACE}/install-{token}', 'pip temporary directory')
        work.mkdir()
        receipt = {'path': str(work), 'freed_bytes': 0, 'status': 'removed'}
        try:
            marker = {'kind': 'autorepro_pip', 'version': 1, 'id': token,
                      'pid': os.getpid(), 'created_at': time.time()}
            (work / OWNER_MARKER).write_text(json.dumps(marker), encoding='utf-8')
            with FileLock(str(work / '.owner.lock')):
                yield work
        finally:
            try:
                checked = workspace_path(root, work.relative_to(root),
                                         'pip temporary cleanup', must_exist=True)
                size = temporary_bytes(checked)
                remove_tree(checked)
                receipt['freed_bytes'] = size
            except (OSError, ValueError) as error:
                receipt.update(status='retained', reason=str(error))
            if on_cleanup is not None:
                # Diagnostic storage must not replace a timeout/cancellation.
                try:
                    on_cleanup(receipt)
                except Exception:
                    pass


def reap_pip_temporary(root, *, dry_run=False, now=None):
    """Caller holds the cache lock. Reap only old, unlocked, marked directories.

    A one-day grace period after abrupt termination gives the process owner
    time to stop orphan children even when the parent's OS lock was released.
    """
    root = workspace_path(root, '.', 'dependency cache root')
    result = {'removed': [], 'retained': [], 'errors': [], 'freed_bytes': 0,
              'remaining_bytes': 0, 'dry_run': dry_run}
    parent = workspace_path(root, TEMP_NAMESPACE, 'pip temporary namespace')
    if not parent.is_dir():
        return result
    now = time.time() if now is None else now
    for directory in sorted(parent.iterdir()):
        identifier = directory.relative_to(root).as_posix()
        try:
            match = _NAME.fullmatch(directory.name)
            checked = workspace_path(root, identifier, 'pip orphan', must_exist=True)
            if not match or not checked.is_dir():
                result['retained'].append({'path': identifier, 'reason': 'unowned'})
                continue
            marker_path = workspace_path(root, checked.relative_to(root) / OWNER_MARKER, 'pip owner marker', must_exist=True)
            marker = json.loads(marker_path.read_text('utf-8'))
            stamp = marker.get('created_at') if isinstance(marker, dict) else None
            if (not isinstance(marker, dict) or marker.get('kind') != 'autorepro_pip'
                    or marker.get('version') != 1 or marker.get('id') != match[1]
                    or type(stamp) not in (int, float) or not math.isfinite(stamp)):
                result['retained'].append({'path': identifier, 'reason': 'unowned'})
                continue
            size = temporary_bytes(checked)
            if now - stamp < ORPHAN_GRACE_SECONDS:
                result['remaining_bytes'] += size
                result['retained'].append({'path': identifier, 'bytes': size, 'reason': 'grace_period'})
                continue
            lock_path = workspace_path(root, checked.relative_to(root) / '.owner.lock', 'pip owner lock')
            lock = FileLock(str(lock_path))
            try:
                lock.acquire(timeout=0)
            except Timeout:
                result['remaining_bytes'] += size
                result['retained'].append({'path': identifier, 'bytes': size, 'reason': 'active'})
                continue
            else:
                # Windows cannot remove a held lock file. The root lock still
                # prevents another installer/collector from taking this path.
                lock.release()
            checked = workspace_path(root, identifier, 'pip orphan cleanup', must_exist=True)
            size = temporary_bytes(checked)
            if not dry_run:
                remove_tree(checked)
            result['removed'].append({'path': identifier, 'bytes': size, 'reason': 'abandoned_install'})
            result['freed_bytes'] += size
        except (OSError, ValueError) as error:
            result['errors'].append({'path': identifier, 'reason': str(error)})
    return result

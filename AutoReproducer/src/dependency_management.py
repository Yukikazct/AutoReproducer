"""Reuse wheel environments and bound the managed cache without touching runs."""
import importlib.metadata
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from src.safety.paths import is_link, workspace_path
from src.safety.files import directory_bytes, remove_tree


def read_metadata(directory):
    try:
        value = json.loads((Path(directory) / 'meta.json').read_text('utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def environment_directories(root):
    """Enumerate environment leaves, including supplemental installs."""
    root = Path(root)
    if not root.is_dir() or is_link(root):
        return
    for path in sorted(root.iterdir()):
        if not path.is_dir() or is_link(path) or path.name.startswith('.'):
            continue
        if path.name != 'repository':
            yield path
            continue
        for runtime in sorted(path.iterdir()):
            if not runtime.is_dir() or is_link(runtime) or runtime.name.startswith('.'):
                continue
            for environment in sorted(runtime.iterdir()):
                if not environment.is_dir() or is_link(environment) or environment.name.startswith('.'):
                    continue
                if environment.name == 'supplemental':
                    for group in sorted(environment.iterdir()):
                        if group.is_dir() and not is_link(group):
                            for leaf in sorted(group.iterdir()):
                                if leaf.is_dir() and not is_link(leaf) and leaf.name.startswith('heal-'):
                                    yield leaf
                else:
                    yield environment


def environment_paths(root, directory, *, require_ready=False):
    """Validate a delta and its single immutable base under the same cache root."""
    root, directory = Path(root).absolute(), Path(directory).absolute()
    relative = directory.relative_to(root)
    if relative == Path('.'):
        raise ValueError('an environment cannot be the cache root')
    directory = workspace_path(root, relative, 'dependency environment', must_exist=True)
    if require_ready and not (directory / '.ready').is_file():
        raise ValueError('dependency environment is incomplete')
    paths = [directory]
    base = read_metadata(directory).get('base_environment')
    if base:
        target = workspace_path(root, base, 'base dependency environment', must_exist=True)
        if (target == directory or target.parent != directory.parent
                or not (target / '.ready').is_file() or read_metadata(target).get('base_environment')):
            raise ValueError('base dependency environment is incomplete or nested')
        paths.append(target)
    return paths


def requirement_signature(requirements):
    """Only plain wheel requirements and reviewed public supplemental indexes."""
    if not isinstance(requirements, str):
        return None
    packages, options = {}, []
    for raw in requirements.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-'):
            if not re.fullmatch(r'--extra-index-url(?:\s+|=)https://download\.pytorch\.org/whl/(?:cpu|cu121)/?', line):
                return None
            options.append(re.sub(r'\s+|=', ' ', line).rstrip('/'))
            continue
        try:
            requirement = Requirement(line)
        except ValueError:
            return None
        if requirement.url or requirement.extras or requirement.marker:
            return None
        name = canonicalize_name(requirement.name)
        spec = str(requirement.specifier)
        if name in packages and packages[name] != spec:
            return None
        packages[name] = spec
    return packages, tuple(sorted(set(options)))


def installed_versions(paths):
    versions = {}
    for path in paths:
        local = {}
        for distribution in importlib.metadata.distributions(path=[str(path)]):
            name = canonicalize_name(distribution.metadata.get('Name', ''))
            if not name or name in local:
                raise ValueError('invalid or duplicate cached distribution metadata')
            local[name] = distribution.version
        for name, version in local.items():
            versions.setdefault(name, version)
    return versions


def compatible_candidates(runtime_root, requirements):
    requested = requirement_signature(requirements)
    if not requested or not requested[0] or not Path(runtime_root).is_dir():
        return []
    candidates = []
    for directory in Path(runtime_root).iterdir():
        if not directory.is_dir() or is_link(directory) or not (directory / '.ready').is_file():
            continue
        signature = requirement_signature(read_metadata(directory).get('requirements', ''))
        if signature and signature[1] == requested[1] and requested[0].items() <= signature[0].items():
            candidates.append((len(signature[0]), str(directory), directory))
    return [candidate[2] for candidate in sorted(candidates)]


def wheel_resolution(report):
    """Use pip's actual resolver and exact wheel hashes, never guess transitive pins."""
    wheels = {}
    for item in report.get('install', []):
        metadata, download = item['metadata'], item['download_info']
        name, version = canonicalize_name(metadata['name']), metadata['version']
        url = download['url']
        parsed = urlsplit(url)
        digest = download.get('archive_info', {}).get('hashes', {}).get('sha256', '')
        if (name in wheels or parsed.scheme != 'https' or parsed.username or parsed.password
                or not parsed.path.lower().endswith('.whl') or not re.fullmatch('[a-fA-F0-9]{64}', digest)):
            raise ValueError('resolver must provide unique HTTPS wheels with SHA-256')
        wheels[name] = {'version': version, 'requirement': name + ' @ ' + url.split('#')[0] + '#sha256=' + digest}
    if not wheels:
        raise ValueError('resolver returned no wheel closure')
    return wheels


def choose_base(runtime_root, requirements, wheels):
    signature = requirement_signature(requirements)
    if not signature:
        return None
    choices = []
    for directory in Path(runtime_root).iterdir():
        meta = read_metadata(directory)
        if (not directory.is_dir() or is_link(directory) or not (directory / '.ready').is_file()
                or meta.get('base_environment')):
            continue
        candidate = requirement_signature(meta.get('requirements', ''))
        if not candidate or candidate[1] != signature[1]:
            continue
        try:
            versions = installed_versions([directory])
        except (OSError, ValueError):
            continue
        shared = {name for name, wheel in wheels.items() if versions.get(name) == wheel['version']}
        if not shared:
            continue
        # Avoid copying native stacks even when another base has more tiny packages.
        score = sum(100 if name in {'torch', 'tensorflow', 'jaxlib'} else
                    20 if name in {'numpy', 'scipy'} else 1 for name in shared)
        choices.append((score, len(shared), str(directory), directory, shared))
    if not choices:
        return None
    chosen = max(choices)
    return chosen[3], {name: wheel for name, wheel in wheels.items() if name not in chosen[4]}, chosen[4]


def prune_cache(root, *, protected=(), quota_bytes=20 * 1024**3,
                min_free_bytes=2 * 1024**3, keep_days=30, dry_run=False):
    """Caller holds the root lock; referenced bases and active paths stay intact."""
    for name, value in (('quota_bytes', quota_bytes), ('min_free_bytes', min_free_bytes),
                        ('keep_days', keep_days)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite non-negative number')
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    protected = {Path(path).resolve() for path in protected}
    rows, errors, unmeasured = [], [], []
    for directory in environment_directories(root):
        try:
            checked = workspace_path(root, directory.relative_to(root), 'dependency eviction', must_exist=True)
            used = checked.stat().st_mtime
        except (OSError, ValueError) as error:
            # Do not hide incomplete accounting from the maintenance receipt.
            name = directory.relative_to(root).as_posix()
            errors.append({'environment': name, 'reason': str(error)})
            unmeasured.append(name)
            continue
        meta = read_metadata(checked)
        try:
            stamp = meta.get('last_used')
            if stamp:
                used = datetime.fromisoformat(stamp).astimezone(timezone.utc).timestamp()
        except (OSError, ValueError, TypeError, OverflowError):
            pass  # Damaged recency metadata falls back to the directory mtime.
        base = None
        if meta.get('base_environment'):
            try:
                base = workspace_path(root, meta['base_environment'], 'base dependency environment')
            except (OSError, ValueError, TypeError):
                pass
        size = 0
        try:
            size = directory_bytes(checked)
        except (OSError, ValueError) as error:
            name = directory.relative_to(root).as_posix()
            errors.append({'environment': name, 'reason': str(error)})
            unmeasured.append(name)
            # An unmeasurable child still owns its base. Keep both in the graph
            # so a link or permission error cannot cause the base to be evicted.
            protected.add(checked)
        rows.append({'path': checked, 'bytes': size, 'used': used,
                     'ready': (checked / '.ready').is_file(), 'base': base})
    remaining = {row['path']: row for row in rows}
    total, free = sum(row['bytes'] for row in rows), shutil.disk_usage(root).free
    now = datetime.now(timezone.utc).timestamp()
    removed = []
    # Re-evaluate references after each child eviction, allowing its base to follow.
    while True:
        referenced = {row['base'] for row in remaining.values() if row['base']}
        eligible = []
        for row in remaining.values():
            if row['path'] in protected or row['path'] in referenced:
                continue
            age = now - row['used']
            cold = age > keep_days * 86400
            incomplete = not row['ready'] and age > 86400
            pressure = total > quota_bytes or free < min_free_bytes
            if cold or incomplete or pressure:
                eligible.append(row)
        if not eligible:
            break
        row = min(eligible, key=lambda item: (item['ready'], item['used']))
        if not dry_run:
            try:
                checked = workspace_path(root, row['path'].relative_to(root), 'dependency eviction', must_exist=True)
                directory_bytes(checked)  # Reject links introduced after inventory.
                remove_tree(checked)
            except (OSError, ValueError) as error:
                errors.append({'environment': row['path'].relative_to(root).as_posix(), 'reason': str(error)})
                protected.add(row['path'])
                continue
        removed.append({'environment': row['path'].relative_to(root).as_posix(), 'bytes': row['bytes']})
        remaining.pop(row['path'])
        total -= row['bytes']
        free = free + row['bytes'] if dry_run else shutil.disk_usage(root).free
    return {'removed': removed, 'errors': errors, 'freed_bytes': sum(row['bytes'] for row in removed),
            'remaining_bytes': total, 'free_bytes': free, 'quota_bytes': quota_bytes,
            'min_free_bytes': min_free_bytes, 'dry_run': dry_run,
            'unmeasured_environments': unmeasured,
            'quota_satisfied': not unmeasured and total <= quota_bytes,
            'space_satisfied': free >= min_free_bytes}

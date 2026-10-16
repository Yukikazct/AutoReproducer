"""Validate committed file aliases inside an immutable Git archive."""
from pathlib import PureWindowsPath
import unicodedata

from src.safety.paths import relative_path, workspace_path


def validated_archive(tar, destination):
    """Return members and regular-file sources without following host links.

    A Git symlink may name another committed regular file. Materialize only
    that file's bytes, retaining the original link in the source manifest.
    Directory links, hard links and every ambiguous namespace fail before I/O.
    """
    members, directories, spellings = {}, set(), {}
    for member in tar.getmembers():
        name = member.name.rstrip('/') if member.isdir() else member.name
        try:
            workspace_path(destination, name, 'archive', forbid_git=True)
            parts = relative_path(name, 'archive', forbid_git=True).parts
            if (name != '/'.join(parts) or any(char in name for char in '\\<>"|?*')
                    or any(ord(char) < 32 for char in name)):
                raise ValueError('nonportable archive name')
        except ValueError as exc:
            raise ValueError('仓库归档包含越界路径或路径别名') from exc
        if not (member.isdir() or member.isfile() or member.issym()):
            raise ValueError(f'仓库归档包含不支持的 symlink 或链接文件: {name}')
        if name in members:
            raise ValueError(f'仓库归档包含重复路径别名: {name}')
        members[name] = member
        for length in range(1, len(parts) + 1):
            prefix = '/'.join(parts[:length])
            key = unicodedata.normalize('NFC', prefix).casefold()
            if key in spellings and spellings[key] != prefix:
                raise ValueError(f'仓库归档包含大小写或 Unicode 路径别名: {name}')
            spellings[key] = prefix
            if length < len(parts) or member.isdir():
                directories.add(prefix)
    for name in directories:
        if name in members and not members[name].isdir():
            raise ValueError(f'仓库归档的文件或 symlink 被用作目录: {name}')

    links = {}
    for name, member in members.items():
        if not member.issym():
            continue
        raw = member.linkname
        windows = PureWindowsPath(raw)
        if (not raw or '\x00' in raw or '\\' in raw or windows.drive or windows.root):
            raise ValueError(f'仓库 symlink 链接目标不是内部相对文件: {name}')
        parts = name.split('/')[:-1]
        segments = raw.split('/')
        for index, segment in enumerate(segments):
            if segment == '..':
                if not parts:
                    raise ValueError(f'仓库 symlink 链接目标越界: {name}')
                parts.pop()
            elif segment not in ('', '.'):
                try:
                    relative_path(segment, 'symlink', forbid_git=True)
                    if (any(char in segment for char in '<>"|?*')
                            or any(ord(char) < 32 for char in segment)):
                        raise ValueError('nonportable target')
                except ValueError as exc:
                    raise ValueError(f'仓库 symlink 链接目标包含路径别名: {name}') from exc
                parts.append(segment)
            # Traversal through file aliases is deliberately unsupported; the
            # archive namespace, rather than the host filesystem, is authority.
            if index < len(segments) - 1 and parts and '/'.join(parts) not in directories:
                raise ValueError(f'仓库 symlink 链接目标经过非目录: {name}')
        target = '/'.join(parts)
        if target not in members or members[target].isdir():
            raise ValueError(f'仓库 symlink 链接目标缺失或不是文件: {name}')
        links[name] = target

    sources = {name: name for name, member in members.items() if member.isfile()}
    aliases = []
    for name in links:
        current, visited = name, set()
        while current in links:
            if current in visited:
                raise ValueError(f'仓库 symlink 链接包含循环: {name}')
            visited.add(current)
            current = links[current]
        if not members[current].isfile():
            raise ValueError(f'仓库 symlink 链接未解析到普通文件: {name}')
        sources[name] = current
        aliases.append({'path': name, 'link_target': members[name].linkname,
                        'resolved_target': current, 'materialization': 'committed_file_copy'})
    return members, sources, aliases

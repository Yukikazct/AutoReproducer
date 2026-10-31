"""Carry verified runtime repairs into a separately observed source candidate."""
import ast
from copy import deepcopy
from pathlib import Path

from src.discovered_repository_runtime import CAPTURE_DIR
from src.method_adapters import digest, read_json, write_json
from src.safety.paths import filesystem_path, workspace_path
from src.safety.workspace_snapshot import iter_workspace_files

COMPATIBILITY_FIELDS = ('numeric_compatibility', 'path_compatibility', 'pyg_compatibility')


def original_observation(workspace, plan, snapshot, spec_sha256):
    """Only a materialized original observer can authorize retained repairs."""
    path = workspace_path(workspace, CAPTURE_DIR + '/config.json', 'original observer')
    if not path.exists():
        return None
    config = read_json(path)
    if (config.get('spec_sha256') != spec_sha256 or config.get('steps') != plan['steps']
            or config.get('source_files') != snapshot['files']
            or config.get('capture') != plan['capture']):
        raise ValueError('源码修复的原始观测配置与冻结作者计划不一致。')
    return config


def observed_input_files(workspace, config):
    """Freeze only previously used data roots and exact recorded PyG inputs."""
    from src.repository_repairer import MAX_REPAIR_WORKSPACE_BYTES, MAX_REPAIR_WORKSPACE_FILES
    from src.experiments.author_path_compatibility import replacement_root, temporary_root
    files = deepcopy(config.get('data_files', {}))
    total = sum(record['bytes'] for record in files.values())

    def add(relative, expected=None):
        nonlocal total
        path = workspace_path(workspace, relative, 'observed repair input', must_exist=True, forbid_git=True)
        size = path.stat().st_size
        if relative not in files:
            total += size
        if len(files) >= MAX_REPAIR_WORKSPACE_FILES or total > MAX_REPAIR_WORKSPACE_BYTES:
            raise ValueError('已观测的修复输入超过隔离副本的文件或字节上限。')
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('已观测的兼容输入发生变化：' + relative)
        record = {'sha256': sha, 'bytes': size}
        if relative in files and any(files[relative].get(key) != value for key, value in record.items()):
            raise ValueError('兼容输入与冻结数据冲突：' + relative)
        files[relative] = record

    for repair in config.get('pyg_compatibility', []):
        add(repair['data_path'], repair['data_sha256'])
    for repair in config.get('path_compatibility', []):
        if (not temporary_root(repair.get('original_root'))
                or repair.get('replacement_root') != replacement_root(repair['original_root'])):
            raise ValueError('原始数据目录兼容凭据无效。')
        root = workspace_path(workspace, repair['replacement_root'], 'observed author data root')
        if root.exists():
            for path in iter_workspace_files(root):
                add(path.relative_to(Path(filesystem_path(workspace))).as_posix())
    return files


def _paired_nodes(original, candidate, relative, source_sha256, effective_sources):
    before = workspace_path(original, relative, 'original compatibility source', must_exist=True)
    after = workspace_path(candidate, relative, 'repaired compatibility source', must_exist=True)
    if digest(before) != source_sha256 or digest(after) != effective_sources.get(relative):
        raise ValueError('兼容修复的原始或候选源码哈希不一致：' + relative)
    old_text, new_text = before.read_text('utf-8'), after.read_text('utf-8')
    old_nodes, new_nodes = list(ast.walk(ast.parse(old_text))), list(ast.walk(ast.parse(new_text)))
    if len(old_nodes) != len(new_nodes) or any(type(a) is not type(b) for a, b in zip(old_nodes, new_nodes)):
        raise ValueError('源码补丁改变了已记录兼容表达式的语法结构：' + relative)
    return old_text, new_text, list(zip(old_nodes, new_nodes))


def _relocate_expression(location, old_text, new_text, pairs):
    matches = [(old, new) for old, new in pairs
               if getattr(old, 'lineno', None) == location.get('line')
               and getattr(old, 'col_offset', None) == location.get('column')
               and ast.get_source_segment(old_text, old) == location.get('original_expression')]
    if len(matches) != 1 or ast.dump(matches[0][0]) != ast.dump(matches[0][1]):
        raise ValueError('源码补丁修改了先前兼容修复所依据的表达式。')
    new = matches[0][1]
    return {**location, 'line': new.lineno, 'column': new.col_offset,
            'original_expression': ast.get_source_segment(new_text, new)}


def rebase_compatibility(config, original, candidate, effective_sources):
    """Bind unchanged AST expressions to candidate hashes and new line numbers.

    Blank lines/formatting may move an expression, but a changed expression
    cannot inherit the earlier error as authorization for a different repair.
    """
    output = {field: deepcopy(config.get(field, [])) for field in COMPATIBILITY_FIELDS}
    for field in ('numeric_compatibility', 'path_compatibility'):
        for index, repair in enumerate(output[field]):
            name = repair['path']
            texts = _paired_nodes(original, candidate, name, repair['source_sha256'], effective_sources)
            if repair.get('kind') == 'numpy_ragged_summary':
                updated = {**repair, 'calls': [_relocate_expression(call, *texts) for call in repair['calls']]}
            else:
                updated = _relocate_expression(repair, *texts)
            updated['source_sha256'] = effective_sources[name]
            output[field][index] = updated
    for repair in output['pyg_compatibility']:
        name = repair['caller_path']
        _, _, pairs = _paired_nodes(original, candidate, name, repair['caller_sha256'], effective_sources)
        statements = [(old, new) for old, new in pairs if isinstance(old, ast.stmt)
                      and old.lineno <= repair['caller_line'] <= old.end_lineno]
        if not statements:
            raise ValueError('无法定位先前 PyG 兼容调用所在语句。')
        old, new = min(statements, key=lambda pair: pair[0].end_lineno - pair[0].lineno)
        if ast.dump(old) != ast.dump(new):
            raise ValueError('源码补丁改变了先前 PyG 兼容调用所在语句。')
        # A multiline expression may be reformatted. Locate its actual access
        # line only when line offsets are unchanged; otherwise fail closed.
        if old.end_lineno - old.lineno != new.end_lineno - new.lineno:
            raise ValueError('PyG 兼容调用的多行位置发生歧义。')
        repair['caller_line'] += new.lineno - old.lineno
        for prefix in ('caller', 'dataset'):
            name = repair[prefix + '_path']
            _paired_nodes(original, candidate, name, repair[prefix + '_sha256'], effective_sources)
            repair[prefix + '_sha256'] = effective_sources[name]
    return output


def materialize_candidate_observation(original, candidate, plan, snapshot, spec_sha256,
                                     config, effective_sources, frozen_data):
    from src.author_execution_plan import materialize_author_execution
    materialized = materialize_author_execution(candidate, plan, spec_sha256,
        {**snapshot, 'files': effective_sources}, retained_files=frozen_data)
    runtime = Path(candidate) / CAPTURE_DIR
    candidate_config = read_json(runtime / 'config.json')
    candidate_config.update(rebase_compatibility(config, original, candidate, effective_sources))
    candidate_config['original_source_files'] = snapshot['files']
    candidate_config['data_files'] = frozen_data
    write_json(runtime / 'config.json', candidate_config)
    materialized['files'][CAPTURE_DIR + '/config.json'] = digest(runtime / 'config.json')
    return materialized

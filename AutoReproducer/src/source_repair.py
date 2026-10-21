"""Explicit compatibility source repair in a separate, fully audited copy.

The pinned author export, observer and frozen plan remain untouched. Repaired
commands are ordinary owned processes, not independent paper verification.
"""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

from src.execution_plan import build_plan
from src.method_adapters import digest, read_json, write_json
from src.safety.patch_policy import PatchPolicy
from src.safety.paths import filesystem_path, relative_path, workspace_path


_SENSITIVE = re.compile(r'(?:train|valid|test|split|seed|epoch|repeat|fold|loss|metric|accura|score|'
                        r'optim|backward|shuffle|sampl|batch|label|target|mask|indices|weight|bias)', re.I)
_PROTOCOL_NAMES = {'acc', 'auc', 'ap', 'map', 'f1', 'precision', 'recall', 'lr', 'mse', 'rmse',
                   'forward', 'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'median', 'std', 'var'}


def _sensitive(name):
    return _SENSITIVE.search(name) or name.casefold() in _PROTOCOL_NAMES


def normalize_source_repair_request(request):
    enabled = request.get('source_repair', False)
    if type(enabled) is not bool:
        raise ValueError('source_repair must be an explicit boolean')
    if not enabled:
        return {**request, 'source_repair': False, 'source_repair_editable': []}
    if (not request.get('pdf_path') or request.get('mock_mode') or request.get('code')
            or request.get('corpus_paper') or request.get('experiment_profile')):
        raise ValueError('源码修复需要普通真实 PDF 与隔离作者仓库，不能用于标题、Mock、外部代码或预设。')
    if request.get('strict_validation'):
        raise ValueError('源码修复与严格核验不能同时启用；修改后执行须单独记录，不能作为原始源码严格验收。')
    values = request.get('source_repair_editable')
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise ValueError('源码修复需要明确列出 1–32 个允许修改的现存 Python 文件。')
    selected, seen = [], set()
    for value in values:
        if not isinstance(value, str) or value != value.strip() or any(c in value for c in '*?[]'):
            raise ValueError('修复白名单必须是精确的相对源码路径，不支持通配符或目录。')
        name = relative_path(value, 'repair source', forbid_git=True).as_posix()
        if Path(name).suffix.lower() != '.py' or not PatchPolicy(editable=[name]).decide(name).allowed:
            raise ValueError('修复白名单不能包含数据、评测、权重或其他受保护文件：' + name)
        if name.casefold() in seen:
            raise ValueError('修复白名单包含重复文件：' + name)
        seen.add(name.casefold())
        selected.append(name)
    return {**request, 'source_repair': True, 'source_repair_editable': selected}


def validate_repair_sources(request, workspace, snapshot):
    if not request.get('source_repair'):
        return
    for name in request['source_repair_editable']:
        if name not in snapshot['files']:
            raise ValueError('允许修复的文件不在固定作者源码中：' + name)
        target = workspace_path(workspace, name, 'repair source', must_exist=True, forbid_git=True)
        if not target.is_file() or digest(target) != snapshot['files'][name]:
            raise ValueError('允许修复的文件已偏离固定源码：' + name)


def _compatibility_shape(source):
    """Permit API spelling compatibility, preserving computation and protocol.

    Variable references, constants, operators, control flow, imports, arguments
    and protocol-sensitive names remain exact. This is deliberately a bounded
    compatibility language, not a claim to decide arbitrary program equivalence.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not _sensitive(node.attr):
            node.attr = '_compatibility_api'
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _sensitive(node.name):
            node.name = '_compatibility_api'
        elif isinstance(node, ast.keyword) and node.arg and not _sensitive(node.arg):
            node.arg = '_compatibility_keyword'
    return ast.dump(tree, include_attributes=False)


def validate_compatibility_changes(original, candidate, files):
    for change in files:
        name = change['path']
        before = workspace_path(original, name, 'original compatibility source', must_exist=True)
        after = workspace_path(candidate, name, 'candidate compatibility source', must_exist=True)
        if _compatibility_shape(before.read_bytes()) != _compatibility_shape(after.read_bytes()):
            raise ValueError('兼容补丁改变了变量、字面量、运算、控制流、依赖或训练/划分/评测结构：' + name)


def validate_compatibility_proposal(original, files, diagnostic):
    """Check the entire set before application, against actual missing API names."""
    missing = set(re.findall(r"(?:has no attribute|unexpected keyword argument) ['\"]([A-Za-z_]\w*)['\"]",
                             diagnostic))
    for change in files:
        name = change['path']
        before = workspace_path(original, name, 'original compatibility source', must_exist=True).read_bytes()
        after = change['content']
        if _compatibility_shape(before) != _compatibility_shape(after):
            raise ValueError('兼容补丁改变了变量、字面量、运算、控制流、依赖或训练/划分/评测结构：' + name)
        for old, new in zip(ast.walk(ast.parse(before)), ast.walk(ast.parse(after))):
            field = ('attr' if isinstance(old, ast.Attribute) else 'name'
                     if isinstance(old, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'arg'
                     if isinstance(old, ast.keyword) else None)
            if field and getattr(old, field) != getattr(new, field):
                if getattr(old, field) not in missing:
                    raise ValueError('补丁修改了实际报错未指向的 API 名称：' + name)


def _frozen_data(provenance):
    files = {}
    for source in provenance.get('sources', []):
        files.update(source.get('extracted_files', {}))
        if source.get('kind') == 'https':
            files[source['target']] = {key: source[key] for key in ('sha256', 'bytes')}
        elif source.get('kind') == 'author_loader' and source.get('verified'):
            files.update(source.get('files', {}))
    return files


def _verify_files(root, expected):
    for name, sha in expected.items():
        path = workspace_path(root, name, 'repair frozen input', must_exist=True, forbid_git=True)
        if not path.is_file() or digest(path) != sha:
            raise ValueError('修复副本的固定输入或已批准源码改变：' + name)


def repair_author_failure(runner, llm, request, plan, snapshot, workspace, run_dir,
                          data, execution, verify_inputs, on_event):
    """Run at most one two-round source repair session after actual execution."""
    state = data['source_repair']
    data.setdefault('original_execution', deepcopy(execution))
    write_json(run_dir / 'original_execution.json', data['original_execution'])
    try:
        return _repair_author_failure(runner, llm, request, plan, snapshot, workspace, run_dir,
                                      data, execution, verify_inputs, on_event)
    except BaseException as exc:
        if state.get('status') != 'rollback_failed':
            state.update(status='failed' if isinstance(exc, Exception) else 'interrupted',
                         reason=str(exc) or type(exc).__name__)
        raise
    finally:
        try:
            write_json(run_dir / 'source_repair.json', state)
            from src.source_repair_evidence import archive_repair_evidence
            archive_repair_evidence(run_dir, data)
            write_json(run_dir / 'source_repair.json', state)
        except Exception as exc:
            state['evidence_error'] = str(exc)
            try:
                write_json(run_dir / 'source_repair.json', state)
            except OSError:
                pass


def _relocate_environment(environment, original, candidate):
    """Preserve author path meaning inside the corresponding repair copy."""
    result, relocated = {}, []
    for key, value in environment.items():
        parts = value.split(os.pathsep) if key == 'PYTHONPATH' else [value]
        values = []
        for part in parts:
            path = Path(part)
            if part and path.is_absolute():
                try:
                    relative = path.relative_to(Path(original).resolve())
                except ValueError:
                    raise ValueError('修复环境包含无法隔离的仓库外绝对路径：' + key)
                mapped = str(workspace_path(candidate, relative.as_posix() or '.', 'relocated author environment'))
                values.append(mapped)
                relocated.append({'name': key, 'original': part, 'replacement': mapped})
            else:
                values.append(part)
        result[key] = os.pathsep.join(values) if key == 'PYTHONPATH' else values[0]
    return result, relocated


def _repair_author_failure(runner, llm, request, plan, snapshot, workspace, run_dir,
                           data, execution, verify_inputs, on_event):
    from src.repository_repairer import (RepositoryRepairer, MAX_REPAIR_WORKSPACE_BYTES,
                                         MAX_REPAIR_WORKSPACE_FILES)
    state = data['source_repair']
    if execution.get('success'):
        state['status'] = 'not_needed'
        write_json(run_dir / 'source_repair.json', state)
        return execution
    if not RepositoryRepairer._repairable(execution):
        state.update(status='not_applicable', reason='安装失败、未运行、取消或超时不触发源码修复。')
        write_json(run_dir / 'source_repair.json', state)
        return execution
    if any(step.get('entrypoint', {}).get('kind') == 'notebook' for step in plan['steps']):
        state.update(status='not_applicable', reason='当前源码兼容修复仅支持原始 Python 命令，不转换 notebook 执行方式。')
        write_json(run_dir / 'source_repair.json', state)
        return execution
    if any(source.get('kind') == 'author_runtime' for source in data['dataset_provenance']['sources']):
        state.update(status='not_applicable', reason='原失败运行的动态数据尚无完整冻结凭据，不能在副本重新下载后当作相同输入修复。')
        return execution
    verify_inputs()
    target = run_dir / 'repair_workspace'
    initial_files = dict(snapshot['files'])
    frozen_data = _frozen_data(data['dataset_provenance'])
    for name, record in frozen_data.items():
        if name in initial_files and initial_files[name] != record['sha256']:
            raise ValueError('冻结源码与数据的路径冲突：' + name)
        initial_files[name] = record['sha256']
    # Preflight using stat before copying or taking an in-memory transaction.
    total = 0
    for name in initial_files:
        path = workspace_path(workspace, name, 'repair copy input', must_exist=True, forbid_git=True)
        total += path.stat().st_size
    if len(initial_files) > MAX_REPAIR_WORKSPACE_FILES or total > MAX_REPAIR_WORKSPACE_BYTES:
        state.update(status='resource_limit', reason='完整修复快照超过 4096 文件或 256 MiB；保留原失败，未请求源码补丁。')
        write_json(run_dir / 'source_repair.json', state)
        return execution
    target.mkdir()
    state.update(status='running', workspace=str(target), original_workspace=str(workspace),
                 original_execution_dir=execution.get('run_dir'), original_source_files=snapshot['files'],
                 initial_copy_files=initial_files, snapshot_bytes=total)
    write_json(run_dir / 'source_repair.json', state)
    for name, expected in initial_files.items():
        source = workspace_path(workspace, name, 'repair copy input', must_exist=True, forbid_git=True)
        destination = workspace_path(target, name, 'repair copy destination', forbid_git=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('复制期间固定输入发生变化：' + name)
        destination.write_bytes(raw)
    steps = deepcopy(plan['steps'])
    original_steps = {step['id']: step for step in data['execution_plan']['steps']}
    state['environment_relocations'] = []
    for step in steps:
        step['env'], relocations = _relocate_environment(
            original_steps[step['id']].get('env', {}), workspace, target)
        state['environment_relocations'].extend({'step': step['id'], **item} for item in relocations)
        for output in step.get('output_directories', []):
            workspace_path(target, output['path'], 'original output directory').mkdir(parents=True, exist_ok=True)
    environment = deepcopy(execution.get('effective_env_config') or data['env_config'])
    repaired_plan = build_plan(profile='source_repair_author_experiment', spec_sha256=data['spec_sha256'],
        workspace=target, repository=plan['repository'], dataset=data['dataset_provenance'], steps=steps)
    state['execution_plan_path'] = str(run_dir / 'source_repair_execution_plan.json')
    write_json(run_dir / 'source_repair_execution_plan.json', repaired_plan)
    current_files = dict(initial_files)
    observed_diagnostics = []

    def validate_patches(_root, files, failure):
        verify_inputs()
        observed_diagnostics.append(str((failure.get('final') or {}).get('stderr') or failure.get('reason') or ''))
        # Every failed round restores the original tree. A later candidate
        # therefore needs the earlier observed compatibility fix as well.
        validate_compatibility_proposal(workspace, files, '\n'.join(observed_diagnostics))

    def verify_candidate(root, files, _execution):
        verify_inputs()
        validate_compatibility_changes(workspace, root, files)
        expected = {**initial_files, **{change['path']: change['sha256'] for change in files}}
        _verify_files(root, expected)
        current_files.clear()
        current_files.update(expected)

    result = None
    try:
        on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'source_repair',
            'status': 'RUNNING', 'detail': '正在独立副本中尝试最多两轮兼容性源码修复；完整原命令将重跑。'}})
        result = runner.run_with_repairs(target, repaired_plan, environment, llm=llm,
            editable=request['source_repair_editable'], max_rounds=2,
            initial_execution=execution, verify_candidate=verify_candidate,
            validate_patches=validate_patches, on_event=on_event)
        state.update(status=result.get('repair_status', 'not_applicable'),
                     attempts=result.get('repair_attempts', []),
                     records_path=result.get('repair_records', ''), report_path=result.get('repair_report', ''))
        state['changed_files'] = [change for attempt in state['attempts'] if attempt['status'] == 'succeeded'
                                  for change in attempt.get('changed_files', [])]
        if any(attempt.get('execution', {}).get('executed') for attempt in state['attempts']):
            data['execution_source'] = 'patched_author_repository'
            data['reproduction_scope'] = 'patched_author_code_execution'
            result['source_modified'] = True
        state['effective_source_files'] = {name: current_files[name] for name in snapshot['files']}
        if result.get('success'):
            verify_inputs()
            _verify_files(target, current_files)
            state['manifest_path'] = str(run_dir / 'source_repair_manifest.json')
            write_json(run_dir / 'source_repair_manifest.json', {
                'scope': 'patched_author_code_execution', 'repository': plan['repository'],
                'original_source_files': snapshot['files'], 'effective_source_files': state['effective_source_files'],
                'changed_files': state['changed_files'], 'spec_sha256': data['spec_sha256'],
                'execution_plan': repaired_plan, 'frozen_input_files': frozen_data})
            state['manifest_sha256'] = digest(run_dir / 'source_repair_manifest.json')
            data['original_execution_plan'] = data['execution_plan']
            data['execution_plan'] = repaired_plan
            data['execution_source'] = 'patched_author_repository'
            data['reproduction_scope'] = 'patched_author_code_execution'
        elif state['status'] != 'rollback_failed':
            _verify_files(target, initial_files)
        return result
    except (KeyboardInterrupt, SystemExit):
        state['status'] = 'interrupted'
        raise
    except Exception as exc:
        state.update(status='failed', reason=str(exc))
        raise
    finally:
        if result is None:
            records = Path(filesystem_path(target.parent / '.autorepro_repository_repairs'))
            for path in sorted(records.glob('*/repairs.json'), key=lambda item: item.stat().st_mtime, reverse=True):
                record = read_json(path)
                state.update(status=record['status'], attempts=record['attempts'], records_path=str(path),
                             report_path=str(path.with_name('repair_report.md')))
                break
        write_json(run_dir / 'source_repair.json', state)


def verify_repaired_execution(plan, data, execution):
    state = data['source_repair']
    if state.get('status') != 'succeeded' or not execution.get('success'):
        raise ValueError('兼容源码修复尚未成功')
    manifest = Path(state['manifest_path'])
    if digest(manifest) != state['manifest_sha256']:
        raise ValueError('已批准的源码修复清单发生变化')
    frozen = read_json(manifest)
    if (frozen['spec_sha256'] != data['spec_sha256'] or frozen['repository'] != plan['repository']
            or frozen['execution_plan'] != data['execution_plan']):
        raise ValueError('源码修复清单与本次冻结实验不一致')
    if read_json(state['execution_plan_path']) != frozen['execution_plan']:
        raise ValueError('修复执行计划发生变化')
    _verify_files(state['workspace'], frozen['effective_source_files'])
    _verify_files(state['workspace'], {name: record['sha256'] for name, record in frozen['frozen_input_files'].items()})
    records = execution.get('steps', [])
    if ([step['id'] for step in records] != [step['id'] for step in plan['steps']]
            or any(not step.get('executed') or step.get('exit_code') != 0 or not step.get('success')
                   or step.get('cancelled') or step.get('timed_out') for step in records)):
        raise ValueError('修复后的完整原命令未正常完成')
    for original, actual in zip(frozen['execution_plan']['steps'], records):
        expected_cwd = workspace_path(state['workspace'], original['cwd'], 'original command cwd', must_exist=True)
        if (actual.get('argv', [])[1:] != original['argv'][1:]
                or Path(actual.get('cwd', '')).resolve() != expected_cwd.resolve()
                or actual.get('kind') != original['kind'] or actual.get('required') != original['required']):
            raise ValueError('修复后的运行记录改变了冻结命令、目录或步骤要求')
    return {'status': 'author_code_completed', 'result_level': 'author_code_completed',
        'execution_status': 'author_code_completed', 'is_reproduced': None,
        'paper_metrics_verified': False, 'protocol_pass': None, 'independent_metrics_pass': None,
        'training_observed': False, 'source_modified': True,
        'author_execution': [],
        'source_repair_manifest_sha256': state['manifest_sha256'],
        'completion_checks': {'all_required_author_commands_returned': True,
                              'owned_processes_finished': True, 'observed_training_updates': False},
        'warnings': ['执行使用明确授权的兼容性源码补丁；未独立核验算法语义、训练更新或论文数值。'],
        'reason': '兼容性源码修改后的完整原命令已正常结束；原始来源保留，尚未判定论文数值复现。'}

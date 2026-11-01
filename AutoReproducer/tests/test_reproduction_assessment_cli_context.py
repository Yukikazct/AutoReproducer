"""Assessment receives frozen author CLI inputs independently of launch examples."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.reproduction_assessment import assess_reproduction
from test_reproduction_assessment import Model, comparison, execution, proposal


def cli_inputs(status='observed'):
    return {
        'version': 1, 'status': status, 'effective_runtime_verified': False,
        'steps': [{
            'id': 'train', 'status': status,
            'argv': ['python', 'main.py', '--dataset', 'pubmed', '--no-cuda'],
            'entrypoint': {'kind': 'script', 'target': 'main.py', 'source_path': 'main.py'},
            'cwd': '.', 'parser': {'source_path': 'main.py', 'source_sha256': 'a' * 64},
            'values': {
                'dataset': {'value': 'pubmed', 'origin': 'explicit', 'flags': ['--dataset'],
                            'explicit_tokens': ['--dataset', 'pubmed'],
                            'declaration': {'line': 12, 'quote': "parser.add_argument('--dataset', default='cora')"},
                            'default_evidence': []},
                'lr': {'value': 0.156, 'origin': 'parser_default', 'flags': ['--lr'],
                       'explicit_tokens': [],
                       'declaration': {'line': 14, 'quote': "parser.add_argument('--lr', type=float, default=0.156)"},
                       'default_evidence': [{'value': 0.156, 'line': 14}]},
                'weight_decay': {'value': 3.544e-5, 'origin': 'parser_default',
                                 'flags': ['--weight_decay'], 'explicit_tokens': [],
                                 'declaration': {'line': 16, 'quote': "parser.add_argument('--weight_decay', type=float, default=3.544e-5)"},
                                 'default_evidence': [{'value': 3.544e-5, 'line': 16}]},
            },
            'limitations': ['Post-parse assignments and config overrides are not verified.'],
        }],
    }


@pytest.mark.parametrize('status', ['observed', 'partial', 'unavailable'])
def test_assessment_receives_cli_defaults_instead_of_inferring_from_wrapper_or_launcher(status):
    source, run = comparison(), execution()
    source['frozen_cli_inputs'] = cli_inputs(status)
    source['comparisons'][0]['settings']['reason'] = (
        '官方 launcher 示例包含 --lr 0.627 --weight_decay 9.38e-6；本次调用省略两项。')
    run['steps'][0]['argv'] = ['python', '-u', '_autorepro_capture/observe_author.py', 'train']
    original = deepcopy((source, run))
    response = proposal()
    response['reason'] = '本次静态 CLI 使用作者 parser 默认值；运行时覆盖尚未核验。'
    model = Model(response)

    result = assess_reproduction(model, source, run)

    prompt, kwargs = model.calls[0]
    evidence = json.loads(prompt.split('已验证证据：\n', 1)[1])
    assert evidence['frozen_cli_inputs'] == source['frozen_cli_inputs']
    assert evidence['frozen_cli_inputs']['steps'][0]['values']['lr']['value'] == 0.156
    assert evidence['frozen_cli_inputs']['steps'][0]['values']['weight_decay']['value'] == 3.544e-5
    assert evidence['execution']['steps'][0]['argv'] == run['steps'][0]['argv']
    assert 'launcher/README 中的示例命令不是本次调用' in prompt
    assert '摘要未跟踪解析后赋值或 config 覆盖' in prompt
    assert '缺失、partial 或 unavailable 的字段保持未知' in prompt
    assert kwargs['task'] == 'reproduction_assessment'
    assert (source, run) == original
    assert result['status'] == 'completed' and result['verdict'] == response['verdict']
    assert result['reason'] == response['reason']
    assert result['metric_assessments'][0]['tolerance'] == response['metric_assessments'][0]['tolerance']
    assert result['metric_assessments'][0]['actual'] == source['comparisons'][0]['actual']
    assert result['metric_assessments'][0]['reference'] == source['comparisons'][0]['reference']
    raw = json.dumps(response, ensure_ascii=False)
    assert result['attempts'][0]['response_excerpt'] == raw
    assert result['attempts'][0]['response_sha256'] == hashlib.sha256(raw.encode('utf-8')).hexdigest()


def test_frozen_cli_context_uses_existing_credential_redaction_without_mutating_input():
    source = comparison()
    source['frozen_cli_inputs'] = cli_inputs()
    source['frozen_cli_inputs']['steps'][0]['limitations'].append(Model.api_key)
    original = deepcopy(source)
    model = Model(proposal())

    result = assess_reproduction(model, source, execution())

    assert result['status'] == 'completed'
    evidence = json.loads(model.calls[0][0].split('已验证证据：\n', 1)[1])
    assert evidence['frozen_cli_inputs']['steps'][0]['limitations'][-1] == '[REDACTED]'
    assert Model.api_key not in model.calls[0][0]
    assert source == original

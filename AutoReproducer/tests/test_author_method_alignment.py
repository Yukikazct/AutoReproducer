"""Static composition evidence never grants a numerical or strict verdict."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.author_method_alignment import align_method_components


ENTRY = '''import argparse
from models import Gate_Classifier

def train(args):
    model = Gate_Classifier(args.base)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='Beta')
    args = parser.parse_args()
    train(args)
'''

MODELS = '''from library import AlphaConv, BetaConv

class Gate:
    def __init__(self, component):
        self.component = component

    def forward(self, x):
        gated = self.component(x)
        return gated

class Gate_Classifier:
    def __init__(self, base):
        if base == 'Alpha':
            self.conv = AlphaConv()
        elif base == 'Beta':
            self.conv = BetaConv()
        self.gate = Gate(self.conv)

    def forward(self, x):
        transformed = self.conv(x)
        rate = self.gate(x)
        result = transformed * rate
        return result
'''


def make_alignment_sources(tmp_path, *, entry=ENTRY, models=MODELS,
                           paper='Our Gate (applied to Alpha, i.e., Gate-Alpha) method.',
                           implementation='Gate_Classifier', label='Gate-Alpha', base='Alpha'):
    """Reusable integration fixture: arguments match align_method_components."""
    workspace = tmp_path / 'author'
    workspace.mkdir(parents=True, exist_ok=True)
    files = {}
    for path, text in [('train.py', entry), ('models.py', models)]:
        (workspace / path).write_bytes(text.encode('utf-8'))
        files[path] = {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    pdf_hash = 'a' * 64
    packet = {'repository': {'url': 'https://github.com/example/official', 'revision': 'b' * 40, 'files': files},
              'pdf': {'sha256': pdf_hash, 'pages': [{'page': 1, 'text': paper,
                      'sha256': hashlib.sha256(paper.encode()).hexdigest()}]}}
    step = {'id': 'train', 'kind': 'train', 'argv': ['python', 'train.py', '--base', base], 'cwd': '.',
            'entrypoint': {'kind': 'script', 'target': 'train.py', 'source_path': 'train.py'}}
    plan = {'steps': [step]}
    record = {'id': 'train:stdout:1', 'step_id': 'train', 'name': 'accuracy', 'value': 0.8,
              'split': 'test', 'source': 'train:stdout:1', 'source_quote': 'Test accuracy: 0.8'}
    reference = {'method': label, 'dataset': 'Example', 'split': 'test'}
    actual = {'method': implementation, 'dataset': 'Example', 'split': 'test'}
    return reference, actual, packet, plan, workspace, record


def test_explicit_composition_is_scoped_static_evidence_only(tmp_path):
    args = make_alignment_sources(tmp_path)
    before = deepcopy(args[:4])
    evidence = align_method_components(*args)
    assert evidence['status'] == 'source_bound_static', evidence
    assert evidence['reference_label'] == 'Gate-Alpha'
    assert evidence['actual_label'] == 'Gate_Classifier'
    assert evidence['max_comparison_status'] == 'reference_only'
    assert evidence['independently_verified'] is False
    assert evidence['effective_runtime_verified'] is False
    assert evidence['binding']['metric_id'] == args[-1]['id']
    assert evidence['binding']['argv'] == args[3]['steps'][0]['argv']
    assert evidence['binding']['source_sha256'] == {path: value['sha256'] for path, value in args[2]['repository']['files'].items()}
    assert evidence['components']['base'] == 'Alpha'
    assert evidence['components']['base_constructor'] == 'AlphaConv'
    assert args[:4] == before
    assert all(item['quote'] for item in evidence['source_evidence'])


@pytest.mark.parametrize('change', [
    lambda r, a, p, plan, w, m: plan['steps'][0]['argv'].__setitem__(-1, 'Beta'),
    lambda r, a, p, plan, w, m: plan['steps'][0]['argv'].extend(['--base', 'Beta']),
    lambda r, a, p, plan, w, m: plan['steps'][0]['argv'].extend(['--bas', 'Beta']),
    lambda r, a, p, plan, w, m: plan['steps'][0]['argv'].__setitem__(2, '--bas'),
    lambda r, a, p, plan, w, m: m.update(step_id='another'),
    lambda r, a, p, plan, w, m: a.update(id='train:stdout:2'),
    lambda r, a, p, plan, w, m: a.update(step_id='another'),
    lambda r, a, p, plan, w, m: m.update(repository_revision='c' * 40),
    lambda r, a, p, plan, w, m: m.update(interpretation_citations=[{'source': 'repository', 'revision': 'c' * 40}]),
    lambda r, a, p, plan, w, m: r.update(pdf_sha256='d' * 64),
    lambda r, a, p, plan, w, m: plan['steps'][0]['entrypoint'].update(source_path='other.py'),
    lambda r, a, p, plan, w, m: p['repository']['files']['models.py'].update(sha256='0' * 64),
    lambda r, a, p, plan, w, m: p['repository']['files']['models.py'].update(text=MODELS + '\n# changed'),
    lambda r, a, p, plan, w, m: p['pdf']['pages'][0].update(text='Gate (applied to Beta, i.e., Gate-Alpha)'),
    lambda r, a, p, plan, w, m: m.update(summary_source={'path': 'other.py', 'source_sha256': 'f' * 64}),
])
def test_another_branch_scope_or_hash_cannot_reuse_a_bridge(tmp_path, change):
    args = make_alignment_sources(tmp_path)
    change(*args)
    assert align_method_components(*args)['status'] == 'unknown'


@pytest.mark.parametrize('paper,label', [
    ('We compare Gate, Alpha and Gate-Alpha.', 'Gate-Alpha'),
    ('Gate (applied to Alpha, i.e., Gate-Alpha+)', 'Gate-Alpha+'),
    ('Gate (applied to Alpha, i.e., Gate-Alpha (PPR))', 'Gate-Alpha (PPR)'),
    ('Gate (applied to Alpha, i.e., Gate-Beta)', 'Gate-Beta'),
    ('Gate (applied to Alpha, i.e., Gate-Alpha)\nGate (applied to Alpha, i.e., Gate-Alpha)', 'Gate-Alpha'),
])
def test_cooccurrence_invented_names_and_distinct_variants_stay_unknown(tmp_path, paper, label):
    args = make_alignment_sources(tmp_path, paper=paper, label=label)
    assert align_method_components(*args)['status'] == 'unknown'


@pytest.mark.parametrize('entry', [
    ENTRY.replace('    train(args)', "    args.base = 'Beta'\n    train(args)"),
    ENTRY.replace('    train(args)', "    setattr(args, 'base', 'Beta')\n    train(args)"),
    ENTRY.replace('    train(args)', "    args.__dict__['base'] = 'Beta'\n    train(args)"),
    ENTRY.replace('    train(args)', '    configure(args)\n    train(args)'),
    ENTRY.replace('    train(args)', '    train(other_args)'),
    ENTRY.replace("    parser.add_argument('--base'", "    parser.set_defaults(base='Beta')\n    parser.add_argument('--base'"),
    ENTRY.replace("type=str", "type=convert"),
    ENTRY.replace("default='Beta'", "default='Beta', action='append'"),
    ENTRY + '\nGate_Classifier = Other\n',
])
def test_overrides_opaque_conversion_and_argument_escape_stay_unknown(tmp_path, entry):
    args = make_alignment_sources(tmp_path, entry=entry)
    assert align_method_components(*args)['status'] == 'unknown'


@pytest.mark.parametrize('models', [
    MODELS.replace('transformed = self.conv(x)', 'transformed = x'),
    MODELS.replace('rate = self.gate(x)', 'rate = x'),
    MODELS.replace('return result', 'result = x\n        return result'),
    MODELS.replace('gated = self.component(x)', 'gated = x'),
    MODELS.replace('self.gate = Gate(self.conv)', 'self.gate = Gate(other)'),
    MODELS.replace('self.gate = Gate(self.conv)', 'self.conv = BetaConv()\n        self.gate = Gate(self.conv)'),
    MODELS.replace('self.component = component', 'self.component = component\n        self.component = Other()'),
    MODELS.replace("        if base == 'Alpha':", "        base = 'Beta'\n        if base == 'Alpha':"),
    MODELS.replace('class Gate_Classifier:', '@decorate\nclass Gate_Classifier:'),
])
def test_unused_or_overwritten_component_cannot_prove_the_method(tmp_path, models):
    args = make_alignment_sources(tmp_path, models=models)
    assert align_method_components(*args)['status'] == 'unknown'


def test_default_selector_is_not_substituted_for_explicit_current_argv(tmp_path):
    args = make_alignment_sources(tmp_path, entry=ENTRY.replace("default='Beta'", "default='Alpha'"))
    args[3]['steps'][0]['argv'] = ['python', 'train.py']
    assert align_method_components(*args)['status'] == 'unknown'


def test_equivalent_equals_option_is_explicit_and_auditable(tmp_path):
    args = make_alignment_sources(tmp_path)
    args[3]['steps'][0]['argv'] = ['python', 'train.py', '--base=Alpha']
    evidence = align_method_components(*args)
    assert evidence['status'] == 'source_bound_static', evidence
    assert evidence['components']['explicit_tokens'] == ['--base=Alpha']


def test_paper_pages_are_never_joined_to_invent_a_relation(tmp_path):
    args = make_alignment_sources(tmp_path)
    pages = ['Our Gate (applied to Alpha,', 'i.e., Gate-Alpha) method.']
    args[2]['pdf']['pages'] = [{'page': i + 1, 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
                             for i, text in enumerate(pages)]
    assert align_method_components(*args)['status'] == 'unknown'

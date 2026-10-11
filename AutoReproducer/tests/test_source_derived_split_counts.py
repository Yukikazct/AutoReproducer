"""Complete source recipes and real runtime census, using invented data only."""
from copy import deepcopy
import hashlib

import pytest

from src.split_counts import (MASK_BODY, RATE_BODY, SAMPLER_BODY, compile_split_rule, expected_from_census)
from src.discovered_repository_runtime import verify_capture
from test_discovered_repository_series import run_series


def sources():
    rate = ("import torch\nimport sampling as du\n"
            "def split(data, split='dense'):\n"
            + '\n'.join('    ' + line for line in RATE_BODY.replace('DENSE', '[0.5, 0.16666666666666666]')
                        .replace('SPARSE', '[0.025, 0.025]').replace('SAMPLER', 'sample').strip().splitlines()) + '\n')
    sampler = ("import torch\ndef index_to_mask(index, size):\n"
               + '\n'.join('    ' + line for line in MASK_BODY.strip().splitlines())
               + '\ndef sample(data, num_classes, percls_trn=20, val_lb=500, Flag=0):\n'
               + '\n'.join('    ' + line for line in SAMPLER_BODY.strip().splitlines()) + '\n')
    return {'rates.py': {'text': rate}, 'sampling.py': {'text': sampler}}


RULE = {'kind': 'stratified_remainder', 'mode': 'dense',
        'rate_source': {'path': 'rates.py', 'function': 'split'},
        'sampler_source': {'path': 'sampling.py', 'function': 'sample'}}


def compile_fixture(files=None, mode='dense'):
    files = files or sources()
    refs = [{'source': 'repository', 'path': path, 'quote': item['text']} for path, item in files.items()]
    return compile_split_rule(RULE, files, selected_mode=mode, references=refs)


def test_literal_counts_are_derived_from_full_labels_including_short_classes():
    rule = compile_fixture()
    assert expected_from_census(rule, [6, 6]) == 4
    # Total 12, class quota 3, validation 2. A class of size 1 exhausts early.
    assert expected_from_census(rule, [1, 11]) == 6


@pytest.mark.parametrize('change', [
    lambda f: f['rates.py'].update(text=f['rates.py']['text'].replace('int(round(u_split[0]', 'int(u_split[0]')),
    lambda f: f['sampling.py'].update(text=f['sampling.py']['text'].replace('rest_index[val_lb:]', 'rest_index[val_lb:val_lb+1]')),
    lambda f: f['sampling.py'].update(text=f['sampling.py']['text'].replace('Flag=0', 'Flag=1')),
    lambda f: f['sampling.py'].update(text=f['sampling.py']['text'].replace('mask[index] = 1', 'mask[index[:1]] = 1')),
    lambda f: f['rates.py'].update(text=f['rates.py']['text'].replace('import sampling as du', 'import other as du')),
])
def test_changed_author_algorithm_never_reuses_the_recipe(change):
    files = sources()
    change(files)
    with pytest.raises((ValueError, SyntaxError)):
        compile_fixture(files)


def test_recipe_needs_actual_selected_mode_and_complete_original_function_citations():
    with pytest.raises(ValueError, match='command'):
        compile_fixture(mode='sparse')
    files = sources()
    with pytest.raises(ValueError, match='complete original'):
        compile_split_rule(RULE, files, selected_mode='dense',
            references=[{'path': path, 'quote': 'not source'} for path in files])


@pytest.mark.parametrize('census', [[], [True, 3], [0, 4], [-1, 4]])
def test_invalid_or_partial_class_census_is_rejected(census):
    with pytest.raises(ValueError):
        expected_from_census(compile_fixture(), census)


def use_recipe(fixture):
    root = fixture['workspace']
    changed = {path: value['text'] for path, value in sources().items()}
    source = (root / 'train.py').read_text('utf-8')
    source = source.replace('import numpy as np', 'import numpy as np\nfrom rates import split')
    source = source.replace("args = parser.parse_args()", "parser.add_argument('--split', default='dense')\nargs = parser.parse_args()")
    source = source.replace('    if run == 1:\n        labels[-1] = 0\n', '')
    source = source.replace('    idx_test = torch.tensor([8, 9, 10, 11], device=device)', '''    class Bundle:
        pass
    data = Bundle()
    data.x, data.y = features, labels
    data.num_classes = len(labels.unique())
    data.num_nodes = len(labels)
    mask = split(data, args.split)
    idx_test = (mask == 2).nonzero().flatten()''')
    changed['train.py'] = source
    for name, source in changed.items():
        path = root / name
        path.write_text(source, encoding='utf-8')
        fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        citation = next((item for item in fixture['proposal']['citations'] if item.get('path') == name), None)
        if citation is None:
            fixture['proposal']['citations'].append({'id': name.removesuffix('.py'), 'source': 'repository', 'path': name, 'quote': source})
        else:
            citation['quote'] = source
    fixture['proposal']['capture']['expected_test_samples'] = deepcopy(RULE)
    fixture['proposal']['capture']['citations'] += ['rates', 'sampling']


def test_native_series_and_fresh_evaluator_derive_counts_from_all_actual_labels(tmp_path):
    fixture, plan, digest, execution = run_series(tmp_path, mutate=use_recipe)
    result = verify_capture(plan, fixture['workspace'], execution, digest)
    assert result['is_reproduced']
    assert result['capture']['test_samples'] == [4, 4, 4]
    assert all(run['split_census'] == [6, 6] for run in result['capture']['runs'])
    assert all(run['split_census'] == [6, 6] for run in result['independent_metrics']['runs'])
    assert plan['capture']['expected_test_samples']['train_fraction'] == .5

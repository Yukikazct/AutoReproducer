"""Original configuration lookup provenance, without running source or getters."""
import hashlib
import pytest

from src.author_configuration import ConfigurationUnsupported, resolve_configured_parameters


CONFIG = "settings = {'small': {'lr': .05, 'alpha': -.2}, 'large': {'lr': .8, 'alpha': .4}}\n"
PROGRAM = '''from parameters import settings
args = helper.parse_args()
def train(depth=10, lr=.01, **kwargs):
    raise AssertionError('Source must not execute')
if __name__ == '__main__':
    if args.test:
        selected = settings
        print(train(**selected[args.dataset]))
'''


def resolve(program=PROGRAM, config=CONFIG, dataset='small', test=True, cli_lr=False):
    files = {name: {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
             for name, text in {'main.py': program, 'parameters.py': config}.items()}
    cli = {'dataset': {'value': dataset}, 'test': {'value': test}}
    if cli_lr:
        cli['lr'] = {'value': .01}
    return resolve_configured_parameters(files, 'main.py', cli, 'a' * 40)


def test_actual_dataset_and_branch_select_original_values_and_function_defaults():
    parameters = resolve()
    assert {key: value['value'] for key, value in parameters.items()} == {'depth': 10, 'lr': .05, 'alpha': -.2}
    assert parameters['lr']['source']['path'] == 'parameters.py'
    assert parameters['lr']['source']['quote'] == '.05'
    assert parameters['lr']['configuration_binding']['call']['quote'] == 'train(**selected[args.dataset])'
    assert "'lr': .05" in parameters['lr']['configuration_binding']['dictionaries'][0]['quote']
    assert resolve(dataset='large')['lr']['value'] == .8
    assert resolve(test=False) == {}


def test_cli_and_function_setting_names_remain_distinct_when_they_collide():
    parameters = resolve(cli_lr=True)
    assert 'lr' not in parameters and parameters['train.lr']['value'] == .05


@pytest.mark.parametrize('program,config', [
    (PROGRAM.replace('settings\n', 'settings\nsettings.update({})\n', 1), CONFIG),
    (PROGRAM.replace('print(train', "selected['small']['lr'] = .9\n        print(train"), CONFIG),
    (PROGRAM.replace('print(train', "for value in range(1):\n            selected['small']['lr'] = .9\n        print(train"), CONFIG),
    (PROGRAM.replace('print(train', 'mutate(selected)\n        print(train'), CONFIG),
    (PROGRAM, 'settings = read_json_configuration()\n'),
    (PROGRAM, "settings = {'small': {'lr': some_property.value}}\n"),
])
def test_unresolved_or_mutated_configuration_cannot_claim_literal_values(program, config):
    with pytest.raises(ConfigurationUnsupported):
        resolve(program, config)


def test_dictionary_alias_mutation_invalidates_the_parent_lookup_as_well():
    program = PROGRAM.replace('selected = settings',
        "selected = settings\n        alias = selected['small']\n        alias['lr'] = .9")
    with pytest.raises(ConfigurationUnsupported):
        resolve(program)


def test_unselected_or_decorated_calls_never_qualify_an_unknown_parameter():
    assert resolve(PROGRAM.replace('def train', '@decorator\ndef train')) == {}
    assert resolve(PROGRAM.replace('args.test:', 'False:')) == {}


def test_duplicate_keyword_expansion_is_rejected_like_the_original_call():
    with pytest.raises(ConfigurationUnsupported, match='duplicate'):
        resolve(PROGRAM.replace('train(**selected', 'train(lr=.9, **selected'))


@pytest.mark.parametrize('mutation', [
    "mutate(selected['small'])",
    "mutate(value=selected['small'])",
    "del selected['small']['lr']",
    "import unknown_module as selected",
    "from unknown_module import selected",
])
def test_mutations_and_unknown_rebindings_cannot_keep_stale_configuration(mutation):
    program = PROGRAM.replace('print(train', mutation + '\n        print(train')
    with pytest.raises(ConfigurationUnsupported):
        resolve(program)


def test_local_call_with_global_configuration_side_effects_cannot_claim_purity():
    program = PROGRAM.replace('if __name__',
        "def mutate():\n    settings['small']['lr'] = .9\nif __name__")
    program = program.replace('selected = settings', 'mutate()\n        selected = settings')
    with pytest.raises(ConfigurationUnsupported):
        resolve(program)


def test_unknown_branch_invalidates_a_possibly_mutated_dictionary():
    program = PROGRAM.replace('selected = settings',
        "selected = settings\n        if unknown_condition:\n            mutate(selected)")
    with pytest.raises(ConfigurationUnsupported):
        resolve(program)

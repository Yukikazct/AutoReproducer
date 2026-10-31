"""Source-only CLI inference preserves author settings and unresolved ambiguity."""
from src.author_cli_inference import infer_cli


PREFIX = 'import argparse\nparser = argparse.ArgumentParser()\n'


def infer(body, *, inventory=(), source_path='train.py', **kwargs):
    return infer_cli(PREFIX + body + '\nargs = parser.parse_args()\n',
                     inventory=inventory, source_path=source_path, **kwargs)


def test_optional_training_settings_and_dataset_defaults_are_not_overridden():
    result = infer("parser.add_argument('--epochs', type=int, default=1000)\n"
                   "parser.add_argument('--seed', type=int, default=42)\n"
                   "parser.add_argument('--dataset', default='Cora')\n")
    assert result['complete'] and result['arguments'] == []
    assert {entry['dest']: entry['value'] for entry in result['defaults']} == {
        'epochs': 1000, 'seed': 42, 'dataset': 'Cora'}
    assert all(entry['line'] > 0 and 'add_argument' in entry['quote'] for entry in result['defaults'])


def test_literal_constants_and_set_defaults_remain_source_evidence():
    result = infer("DEFAULT_EPOCHS = 350\n"
                   "parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)\n"
                   "parser.add_argument('--dataset', default='original')\n"
                   "parser.set_defaults(dataset='actual_author_default')\n")
    assert result['complete'] and result['arguments'] == []
    assert {entry['dest']: entry['value'] for entry in result['defaults']} == {
        'epochs': 350, 'dataset': 'actual_author_default'}
    assert any('set_defaults' in entry.get('quote', '') for entry in result['evidence'])


def test_required_values_need_literal_defaults_or_one_author_choice():
    result = infer("parser.add_argument('--dataset', required=True, choices=['Cora'])\n"
                   "parser.add_argument('--seed', required=True, type=int, default=7)\n")
    assert result['complete']
    assert result['arguments'] == ['--dataset', 'Cora', '--seed', '7']
    result = infer("parser.add_argument('--dataset', required=True, choices=['Cora', 'PubMed'])\n")
    assert not result['complete'] and result['missing'] == ['--dataset']
    assert result['arguments'] == []


def test_required_config_uses_unique_checked_in_project_config():
    result = infer("parser.add_argument('--config', required=True)\n",
                   source_path='experiments/cora/train.py',
                   inventory=['experiments/cora/configs/author.yaml', 'experiments/pubmed/configs/author.yaml'])
    assert result['complete']
    assert result['arguments'] == ['--config', 'configs/author.yaml']
    assert result['config_paths'] == ['experiments/cora/configs/author.yaml']
    assert result['evidence'][-1]['kind'] == 'configuration_inventory'


def test_config_arguments_resolve_against_explicit_execution_directory():
    result = infer("parser.add_argument('--config', required=True)\n",
                   source_path='experiments/train.py', inventory=['experiments/config.yaml'], cwd='.')
    assert result['arguments'] == ['--config', 'experiments/config.yaml']
    assert result['config_paths'] == ['experiments/config.yaml']


def test_multiple_configs_require_a_unique_prose_preference():
    inventory = ['configs/cora.yaml', 'configs/pubmed.yaml']
    unknown = infer("parser.add_argument('--cfg', required=True)\n", inventory=inventory)
    assert not unknown['complete'] and unknown['missing'] == ['--cfg']
    selected = infer("parser.add_argument('--cfg', required=True)\n", inventory=inventory,
                     preferred_configs=['configs/pubmed.yaml'])
    assert selected['complete'] and selected['arguments'] == ['--cfg', 'configs/pubmed.yaml']
    both = infer("parser.add_argument('--cfg', required=True)\n", inventory=inventory,
                 preferred_configs=inventory)
    assert not both['complete']


def test_arbitrary_dataset_or_package_metadata_is_not_misidentified_as_config():
    result = infer("parser.add_argument('--config', required=True)\n",
                   inventory=['dataset.json', 'labels.json', 'package.json', 'pyproject.toml'])
    assert not result['complete'] and result['arguments'] == []


def test_default_config_is_recorded_without_adding_a_cli_override():
    result = infer("parser.add_argument('--config', default='configs/paper.yaml')\n",
                   inventory=['configs/paper.yaml'])
    assert result['complete'] and result['arguments'] == []
    assert result['config_paths'] == ['configs/paper.yaml']


def test_required_config_defaults_must_exist_under_the_selected_working_directory():
    body = "parser.add_argument('--config', required=True, default='configs/paper.yaml')\n"
    present = infer(body, inventory=['configs/paper.yaml'])
    assert present['complete'] and present['arguments'] == ['--config', 'configs/paper.yaml']
    assert present['config_paths'] == ['configs/paper.yaml']
    absent = infer(body, inventory=['configs/unrelated.yaml'])
    assert not absent['complete'] and absent['arguments'] == []
    escaped = infer("parser.add_argument('--config', required=True, default='../../outside.yaml')\n",
                    source_path='src/train.py', inventory=['outside.yaml'])
    assert not escaped['complete'] and escaped['arguments'] == []


def test_required_positionals_are_not_ignored_but_optional_ones_keep_defaults():
    result = infer("parser.add_argument('dataset', choices=['Cora'])\n"
                   "parser.add_argument('output', nargs='?', default='results')\n")
    assert result['complete'] and result['arguments'] == ['Cora']
    unknown = infer("parser.add_argument('dataset')\n")
    assert not unknown['complete'] and unknown['missing'] == ['dataset']


def test_only_invoked_function_parsers_are_considered():
    source = ("from argparse import ArgumentParser as Parser\n"
              "def unused():\n    p = Parser()\n    p.add_argument('--unrelated', required=True)\n"
              "    return p.parse_args()\n"
              "def get_args():\n    p = Parser()\n    p.add_argument('--dataset', default='Cora')\n"
              "    return p.parse_args()\n"
              "def main():\n    args = get_args()\n    train(args)\n"
              "if __name__ == '__main__':\n    main()\n")
    result = infer_cli(source, inventory=[], source_path='train.py')
    assert result['complete'] and result['missing'] == []
    assert [entry['dest'] for entry in result['defaults']] == ['dataset']
    assert infer_cli(source.split("if __name__")[0], inventory=[], source_path='train.py') is None


def test_import_aliases_and_argument_groups_are_supported():
    source = ("import argparse as cli\np = cli.ArgumentParser()\n"
              "group = p.add_argument_group('training')\n"
              "group.add_argument('--epochs', default=200, type=int)\n"
              "args = p.parse_known_args()\n")
    result = infer_cli(source, inventory=[], source_path='main.py')
    assert result['complete'] and result['defaults'][0]['value'] == 200


def test_dynamic_parser_configuration_cannot_claim_a_complete_command():
    for body in (
        "configure(parser)\n",
        "alias = parser\nalias.configure()\n",
        "parser.add_subparsers()\n",
        "parser.add_argument('--dataset', required=condition)\n",
        "parser.add_argument(*unknown_options)\n",
    ):
        result = infer(body)
        assert not result['complete'], body


def test_author_code_is_parsed_without_being_executed(tmp_path):
    target = tmp_path / 'must-not-exist'
    source = PREFIX + f"open({str(target)!r}, 'w').write('executed')\n" + 'args = parser.parse_args()\n'
    result = infer_cli(source, inventory=[], source_path='train.py')
    assert result['complete'] and not target.exists()


def test_parse_with_an_explicit_argument_list_is_not_inferred_as_process_cli():
    result = infer_cli(PREFIX + "args = parser.parse_args(['--help'])\n",
                       inventory=[], source_path='train.py')
    assert not result['complete']


def test_parser_mutations_after_parsing_do_not_change_the_invocation_snapshot():
    source = PREFIX + ("args = parser.parse_args()\n"
                       "parser.add_argument('--unrelated', required=True)\n")
    result = infer_cli(source, inventory=[], source_path='train.py')
    assert result['complete'] and result['missing'] == []


def test_multiple_parsers_do_not_claim_one_combined_cli_is_unambiguous():
    source = PREFIX + ("args = parser.parse_args()\n"
                       "other = argparse.ArgumentParser()\nother.parse_args()\n")
    result = infer_cli(source, inventory=[], source_path='train.py')
    assert not result['complete']
    assert '<multiple invoked argparse parsers>' in result['missing']


def test_rebound_parser_or_import_alias_cannot_authorize_original_defaults():
    rebound_parser = infer('parser = custom_parser()\n')
    assert not rebound_parser['complete']
    for prefix in ('argparse = fake_module()\n', 'argparse.ArgumentParser = replacement\n'):
        source = 'import argparse\n' + prefix + 'parser = argparse.ArgumentParser()\nparser.parse_args()\n'
        assert infer_cli(source, inventory=[], source_path='train.py') is None


def test_implicit_argparse_defaults_allow_reading_the_actual_selected_branch():
    result = infer("parser.add_argument('--train', action='store_true')\n"
                   "parser.add_argument('--validate', action='store_false')\n"
                   "parser.add_argument('--verbose', action='count')\n"
                   "parser.add_argument('--mode')\n")
    assert result['complete'] and result['arguments'] == []
    assert {entry['dest']: entry['value'] for entry in result['defaults']} == {
        'train': False, 'validate': True, 'verbose': None, 'mode': None}
    assert all(entry['implicit'] for entry in result['defaults'])


def test_string_defaults_follow_builtin_argparse_types_without_calling_author_code():
    result = infer("parser.add_argument('--train', type=int, default='0')\n"
                   "parser.add_argument('--rate', type=float, default='0.1')\n"
                   "parser.add_argument('--custom', type=author_converter, default='original')\n")
    assert result['complete']
    assert {entry['dest']: entry['value'] for entry in result['defaults']} == {'train': 0, 'rate': 0.1}
    assert not infer("parser.add_argument('--train', type=int, default='invalid')\n")['complete']


def test_unknown_explicit_dest_does_not_invent_a_namespace_field():
    result = infer("parser.add_argument('--train', dest=unknown_field, default=True)\n")
    assert not result['complete'] and result['missing'] == ['--train']

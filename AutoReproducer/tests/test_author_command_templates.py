"""README placeholders need explicit, entrypoint-bound finite author values."""
import pytest

from src.discovered_repository_plan import PlanEvidenceError, _enumerated_command_values, _verify_author_arguments


DOCUMENTED = '''
### Synthetic experiments
python synthetic.py --dataset $dataset
where $dataset is selected from toy, other.
### Classification experiments
python classifier.py --repeat 10 --dataset $dataset
where $dataset is selected from cora, citeseer, pubmed, and texas.
'''
PACKET = {'repository': {'files': {'options.py': {'text': '''
parser.add_argument('--dataset', type=str)
parser.add_argument('--repeat', type=int, default=1)
'''}}}}


def test_only_literal_enumeration_for_the_actual_entrypoint_is_compiled():
    assert _enumerated_command_values(DOCUMENTED, 'classifier.py') == {
        '--dataset': {'cora', 'citeseer', 'pubmed', 'texas'}}
    _verify_author_arguments(['--repeat', '10', '--dataset', 'texas'], DOCUMENTED, PACKET, script='classifier.py')


@pytest.mark.parametrize('value', ['toy', 'unknown', 'Texas', '$dataset', 'texas;echo'])
def test_unlisted_or_cross_experiment_values_cannot_fill_a_placeholder(value):
    with pytest.raises(PlanEvidenceError):
        _verify_author_arguments(['--dataset', value], DOCUMENTED, PACKET, script='classifier.py')


@pytest.mark.parametrize('change', [
    lambda text: text.replace('where $dataset is selected from cora, citeseer, pubmed, and texas.',
                              'any dataset can be used'),
    lambda text: text.replace('where $dataset is selected from cora, citeseer, pubmed, and texas.',
                              '### Other experiment\nwhere $dataset is selected from cora, texas.'),
    lambda text: text.replace('where $dataset is selected from cora, citeseer, pubmed, and texas.',
                              'where $dataset is selected from all available datasets.'),
    lambda text: text.replace('python classifier.py', 'python different.py'),
])
def test_ambiguous_prose_or_another_protocol_is_not_a_documented_value(change):
    with pytest.raises(PlanEvidenceError):
        _verify_author_arguments(['--dataset', 'texas'], change(DOCUMENTED), PACKET, script='classifier.py')

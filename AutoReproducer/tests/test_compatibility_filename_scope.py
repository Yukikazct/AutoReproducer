import pytest

from src.repository_runner import RepositoryRunner
from src.safety.patch_policy import PatchPolicy
from src.source_repair import validate_compatibility_proposal
from src.repository_repairer import _introduced_danger


@pytest.mark.parametrize('name', ['worker_eval.py', 'test_runtime_helper.py', 'dataset_helpers.py'])
def test_filename_exception_needs_explicit_scope_and_keeps_ast_validation(tmp_path, name):
    assert not PatchPolicy(editable=[name]).decide(name).allowed
    policy = PatchPolicy(editable=[name], compatibility_source_files=[name])
    assert policy.decide(name).allowed
    source = 'result = dict(answer=7).iteritems()\n'
    (tmp_path / name).write_text(source)
    diagnostic = "AttributeError: 'dict' object has no attribute 'iteritems'"
    validate_compatibility_proposal(tmp_path, [{'path': name, 'content': source.replace('iteritems', 'items')}], diagnostic)
    with pytest.raises(ValueError):
        validate_compatibility_proposal(tmp_path, [{'path': name, 'content': source.replace('answer=7', 'answer=8')}], diagnostic)


@pytest.mark.parametrize('name', ['data/helper.py', 'tests/helper.py', 'evaluation/helper.py', '.git/config.py', 'input.npz'])
def test_filename_scope_cannot_relax_directory_or_data_protections(name):
    policy = PatchPolicy(editable=[name], compatibility_source_files=[name])
    assert not policy.decide(name).allowed


def test_runner_requires_semantic_validators_for_filename_exception(tmp_path):
    with pytest.raises(ValueError, match='validators'):
        RepositoryRunner().run_with_repairs(tmp_path, [], {}, llm=None,
            editable=['worker_eval.py'], compatibility_source_files=['worker_eval.py'])


@pytest.mark.parametrize('operation', ['eval("payload")', 'obj.eval(data)', 'os.system("cmd")', 'shutil.rmtree("data")'])
def test_existing_author_operations_do_not_hide_new_dangerous_calls(operation):
    baseline = 'evaluator.eval(predictions)\n'
    assert _introduced_danger(baseline, baseline + operation) is not None

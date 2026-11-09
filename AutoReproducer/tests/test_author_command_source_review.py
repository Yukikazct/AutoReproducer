"""Incomplete README instructions trigger bounded source reasoning, never execution."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.author_command_inference import discover_author_commands, select_author_command
from src.author_execution_plan import propose_author_execution_plan
from src.discovered_repository_plan import PlanEvidenceError
from test_author_command_inference import TRAIN, packet_for


def execution_payload(packet, argv=None):
    return {'version': 1, 'scope': 'author_code_execution',
        'repository': {key: packet['repository'][key] for key in ('url', 'revision')},
        'device': 'cpu', 'requirements': {'author': []}, 'datasets': [],
        'steps': [{'id': 'train', 'kind': 'train', 'argv': argv or ['python', 'baseline.py'],
                   'cwd': '.', 'timeout_s': None}]}


def imported_entrypoint_packet(tmp_path):
    helper = 'engine/core/trainer.py'
    sources = {
        'README.md': 'The baseline implementation uses the experiment engine and its original settings.',
        'baseline.py': ('from engine.core.trainer import run_experiment\n'
                        "if __name__ == '__main__':\n    run_experiment()\n"),
        helper: ("raise RuntimeError('Planning must not import the experiment engine')\n"
                 'def run_experiment():\n    estimator.fit(records)\n'),
    }
    packet = packet_for(tmp_path, sources)
    packet['repository']['files'].pop(helper)
    return packet, helper, sources[helper]


class UnsupportedPlanner:
    def __init__(self):
        self.calls = []

    def chat(self, prompt, **kwargs):
        self.calls.append(prompt)
        assert kwargs['task'] == 'author_execution_plan'
        return json.dumps({'unsupported': 'README does not provide a complete training command'})


@pytest.mark.parametrize('name', ['baseline.py', 'examples/simple_example.py'])
def test_nonstandard_script_names_expose_original_training_without_readme_commands(tmp_path, name):
    packet = packet_for(tmp_path, {
        'README.md': 'Examples demonstrate the algorithm with its original settings.',
        name: ('import learner\n'
               "if __name__ == '__main__':\n    learner.fit(records, iterations=20)\n"),
    })
    discovery = discover_author_commands(packet, tmp_path)
    candidate = select_author_command(discovery)
    assert candidate['source_path'] == name
    assert candidate['source_sha256'] == packet['file_inventory'][name]['sha256']
    assert 'iterations=20' in candidate['training_evidence'][0]['quote']
    assert not (tmp_path / '__pycache__').exists()


def test_readme_prose_can_identify_a_deep_nonstandard_original_example(tmp_path):
    name = 'demo/guide-python/basic_walkthrough.py'
    packet = packet_for(tmp_path, {
        'README.md': f'The basic example is implemented in `{name}`; use the bundled data.',
        name: "import learner\nlearner.train(parameters, data, num_boost_round=2)\n",
    })
    candidate = select_author_command(discover_author_commands(packet, tmp_path))
    assert candidate['source_path'] == name
    assert candidate['argv'] == ['python', 'basic_walkthrough.py']
    assert candidate['cwd'] == 'demo/guide-python'
    assert candidate['readme_evidence'][0]['quote'].startswith('The basic example')


def test_readme_links_prioritize_real_nested_example_over_unrelated_training_files(tmp_path):
    name = 'demo/guide-python/basic_walkthrough.py'
    sources = {
        'README.md': 'See the [resources](demo/README.md) for examples.',
        'demo/README.md': 'The [Python guide](guide-python/) introduces the algorithm.',
        name: 'import learner\nlearner.train(parameters, data, num_boost_round=2)\n',
        'train.py': TRAIN,
        **{f'unrelated/{index:02d}/README.md': 'Unrelated project documentation.'
           for index in range(10)},
    }
    packet = packet_for(tmp_path, sources)
    for path in list(packet['repository']['files']):
        if path != 'README.md':
            packet['repository']['files'].pop(path)
    discovery = discover_author_commands(packet, tmp_path, max_files=1)
    candidate = select_author_command(discovery)
    assert candidate['source_path'] == name
    assert candidate['source_sha256'] == packet['file_inventory'][name]['sha256']
    assert discovery['files_examined'] == 1
    assert 'demo/README.md' in packet['repository']['files']
    assert 'train.py' not in packet['repository']['files']
    assert sum(path.rsplit('/', 1)[-1] == 'README.md'
               for path in packet['repository']['files']) <= 8


@pytest.mark.parametrize('target', [
    '../outside', '/outside', 'C:/outside', 'https://example.invalid/examples',
])
def test_readme_links_cannot_promote_external_or_unverified_deep_sources(tmp_path, target):
    packet = packet_for(tmp_path, {
        'README.md': f'Follow the [examples]({target}) to learn the algorithm.',
        'research/private/original_example.py': 'import learner\nlearner.fit(records)\n',
    })
    discovery = discover_author_commands(packet, tmp_path)
    assert discovery['candidates'] == [] and discovery['review_candidates'] == []


def test_changed_linked_readme_does_not_establish_an_example_directory(tmp_path):
    packet = packet_for(tmp_path, {
        'README.md': 'See the [resources](demo/README.md).',
        'demo/README.md': 'See the [guide](guide-python/).',
        'demo/guide-python/basic_walkthrough.py': 'import learner\nlearner.fit(records)\n',
    })
    packet['repository']['files'].pop('demo/README.md')
    (tmp_path / 'demo/README.md').write_text('See the changed [guide](guide-python/).', encoding='utf-8')
    discovery = discover_author_commands(packet, tmp_path)
    assert discovery['candidates'] == []
    assert 'demo/README.md' not in packet['repository']['files']


def test_linked_documentation_and_example_share_total_source_byte_budget(tmp_path):
    sources = {
        'README.md': 'See the [resources](demo/README.md).',
        'demo/README.md': 'See the [guide](guide-python/).',
        'demo/guide-python/basic_walkthrough.py': 'import learner\nlearner.fit(records)\n',
    }
    packet = packet_for(tmp_path, sources)
    budget = len(sources['README.md'].encode()) + len(sources['demo/README.md'].encode())
    discovery = discover_author_commands(packet, tmp_path, max_bytes=budget)
    assert discovery['bytes_examined'] == budget
    assert discovery['candidates'] == []


def test_unknown_training_branch_remains_navigation_and_cannot_automatically_run(tmp_path):
    source = ("import argparse\np = argparse.ArgumentParser()\n"
              "p.add_argument('--mode', default='train')\nargs = p.parse_args()\n"
              "if args.mode in ('train', 'fit'):\n    estimator.fit(records)\n")
    packet = packet_for(tmp_path, {'README.md': 'See the original source for available modes.',
                                  'baseline.py': source})
    discovery = discover_author_commands(packet, tmp_path)
    assert discovery['candidates'] == []
    assert select_author_command(discovery) is None
    candidate, = discovery['review_candidates']
    assert candidate['source_path'] == 'baseline.py'
    assert candidate['review_only'] is True and candidate['complete'] is False
    assert 'training behavior requires source review' in candidate['missing']
    with pytest.raises(PlanEvidenceError, match='No unique complete source-derived'):
        propose_author_execution_plan(None, packet, tmp_path)


def test_imported_experiment_is_review_only_until_model_reads_actual_helper(tmp_path):
    packet, helper, _ = imported_entrypoint_packet(tmp_path)
    discovery = discover_author_commands(packet, tmp_path)
    candidate, = discovery['review_candidates']
    assert candidate['related_sources'] == [helper]
    assert candidate['source_sha256'] == packet['file_inventory']['baseline.py']['sha256']
    assert discovery['candidates'] == [] and select_author_command(discovery) is None
    assert helper not in packet['repository']['files']
    with pytest.raises(PlanEvidenceError, match='No unique complete source-derived'):
        propose_author_execution_plan(None, packet, tmp_path)
    assert not (tmp_path / 'engine/core/__pycache__').exists()


def test_shallow_helper_body_is_not_published_by_entrypoint_scanning(tmp_path):
    packet = packet_for(tmp_path, {
        'README.md': 'The baseline uses the original experiment engine.',
        'baseline.py': "from author_model import Experiment\nif __name__ == '__main__':\n    Experiment()\n",
        'author_model.py': 'class Experiment:\n    def fit(self):\n        estimator.fit(records)\n',
    })
    packet['repository']['files'].pop('author_model.py')
    discovery = discover_author_commands(packet, tmp_path)
    candidate, = discovery['review_candidates']
    assert candidate['source_path'] == 'baseline.py'
    assert candidate['related_sources'] == ['author_model.py']
    assert 'author_model.py' not in packet['repository']['files']


def test_unsupported_readme_response_gets_pinned_import_context_before_corrected_plan(tmp_path):
    packet, helper, helper_source = imported_entrypoint_packet(tmp_path)
    before = {name: item['sha256'] for name, item in packet['file_inventory'].items()}

    class CorrectingPlanner(UnsupportedPlanner):
        def chat(self, prompt, **kwargs):
            if not self.calls:
                assert 'estimator.fit(records)' not in prompt
                return super().chat(prompt, **kwargs)
            self.calls.append(prompt)
            assert kwargs['task'] == 'author_execution_plan'
            assert 'Additional pinned source files have now been read' in prompt
            assert 'A README need not contain an executable command' in prompt
            assert helper in prompt and 'estimator.fit(records)' in prompt
            return json.dumps(execution_payload(packet))

    planner = CorrectingPlanner()
    plan = propose_author_execution_plan(planner, packet, tmp_path)
    assert len(planner.calls) == 2
    assert plan['steps'][0]['argv'] == ['python', 'baseline.py']
    assert packet['repository']['files'][helper]['text'] == helper_source
    reviews = packet['program_detected_author_commands']['source_review_reads']
    assert reviews == [[{'path': helper, 'sha256': before[helper]}]]
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == value
               for name, value in before.items())
    assert not (tmp_path / 'engine/core/__pycache__').exists()


def test_repeated_unsupported_does_not_promote_review_candidate_to_fallback(tmp_path):
    packet, helper, _ = imported_entrypoint_packet(tmp_path)
    planner = UnsupportedPlanner()
    with pytest.raises(PlanEvidenceError, match='README does not provide'):
        propose_author_execution_plan(planner, packet, tmp_path)
    assert len(planner.calls) == 2
    assert helper in packet['repository']['files']
    assert len(packet['program_detected_author_commands']['source_review_reads']) == 1
    assert packet['program_detected_author_commands']['candidates'] == []


def nine_helper_packet(tmp_path):
    helpers = [f'engine/core/helper_{index}.py' for index in range(9)]
    sources = {'README.md': 'The original experiments use baseline.py and supplement.py.',
        'baseline.py': ''.join(f'from engine.core.helper_{index} import run_stage_{index}\n'
                               for index in range(8))
                       + "if __name__ == '__main__':\n    run_stage_0()\n",
        'supplement.py': ("from engine.core.helper_8 import run_stage_8\n"
                          "if __name__ == '__main__':\n    run_stage_8()\n")}
    for index, name in enumerate(helpers):
        sources[name] = (f'ORIGINAL_STAGE_{index} = {index}\n'
                         f'def run_stage_{index}():\n    estimator.fit(records)\n')
    packet = packet_for(tmp_path, sources)
    for name in helpers:
        packet['repository']['files'].pop(name)
    return packet, helpers


def test_second_new_helper_batch_reaches_model_after_identical_unsupported_responses(tmp_path):
    packet, helpers = nine_helper_packet(tmp_path)

    class CorrectingPlanner(UnsupportedPlanner):
        def chat(self, prompt, **kwargs):
            if len(self.calls) < 2:
                return super().chat(prompt, **kwargs)
            self.calls.append(prompt)
            assert kwargs['task'] == 'author_execution_plan'
            assert all(f'ORIGINAL_STAGE_{index} = {index}' in prompt for index in range(9))
            return json.dumps(execution_payload(packet))

    planner = CorrectingPlanner()
    plan = propose_author_execution_plan(planner, packet, tmp_path)
    assert len(planner.calls) == 3
    assert plan['steps'][0]['argv'] == ['python', 'baseline.py']
    reviews = packet['program_detected_author_commands']['source_review_reads']
    assert [[record['path'] for record in batch] for batch in reviews] == [helpers[:8], helpers[8:]]
    assert 'ORIGINAL_STAGE_8 = 8' not in planner.calls[1]
    assert all(name in packet['repository']['files'] for name in helpers)
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
               == packet['file_inventory'][name]['sha256'] for name in packet['file_inventory'])
    assert not (tmp_path / 'engine/core/__pycache__').exists()


def test_identical_unsupported_without_new_source_evidence_still_stops_twice(tmp_path):
    packet, helper, source = imported_entrypoint_packet(tmp_path)
    packet['repository']['files'][helper] = {'text': source,
        'bytes': packet['file_inventory'][helper]['bytes'],
        'sha256': packet['file_inventory'][helper]['sha256']}
    planner = UnsupportedPlanner()
    with pytest.raises(PlanEvidenceError, match='README does not provide'):
        propose_author_execution_plan(planner, packet, tmp_path)
    assert len(planner.calls) == 2
    assert not packet['program_detected_author_commands'].get('source_review_reads')


def test_failed_second_helper_batch_does_not_extend_identical_unsupported(tmp_path, monkeypatch):
    from src import author_execution_plan as planning
    packet, helpers = nine_helper_packet(tmp_path)
    (tmp_path / helpers[-1]).write_text('CHANGED_UNVERIFIED_SOURCE = True\n', encoding='utf-8')
    attempted = []
    original_reader = planning._read_planning_sources

    def observe_read(names, *args, **kwargs):
        attempted.append(list(names))
        return original_reader(names, *args, **kwargs)

    monkeypatch.setattr(planning, '_read_planning_sources', observe_read)
    planner = UnsupportedPlanner()
    with pytest.raises(PlanEvidenceError, match='README does not provide'):
        propose_author_execution_plan(planner, packet, tmp_path)
    assert len(planner.calls) == 2
    assert attempted == [helpers[:8], helpers[8:]]
    reviews = packet['program_detected_author_commands']['source_review_reads']
    assert [[record['path'] for record in batch] for batch in reviews] == [helpers[:8]]
    assert helpers[-1] not in packet['repository']['files']
    assert all('CHANGED_UNVERIFIED_SOURCE' not in prompt for prompt in planner.calls)


@pytest.mark.parametrize('budget', ['source_files', 'source_bytes'])
def test_automatic_source_review_respects_original_packet_read_budget(tmp_path, budget):
    packet, helper, _ = imported_entrypoint_packet(tmp_path)
    files = packet['repository']['files']
    packet['limits'][budget] = len(files) if budget == 'source_files' else sum(
        source['bytes'] for source in files.values())
    planner = UnsupportedPlanner()
    with pytest.raises(PlanEvidenceError):
        propose_author_execution_plan(planner, packet, tmp_path)
    assert helper not in packet['repository']['files']
    assert all('estimator.fit(records)' not in prompt for prompt in planner.calls)
    assert not packet['program_detected_author_commands'].get('source_review_reads')


def test_changed_imported_helper_never_becomes_planning_evidence(tmp_path):
    packet, helper, _ = imported_entrypoint_packet(tmp_path)
    (tmp_path / helper).write_text('def run_experiment():\n    changed_source()\n', encoding='utf-8')
    planner = UnsupportedPlanner()
    with pytest.raises(PlanEvidenceError):
        propose_author_execution_plan(planner, packet, tmp_path)
    assert helper not in packet['repository']['files']
    assert not packet['program_detected_author_commands'].get('source_review_reads')
    assert 'changed after snapshot' in planner.calls[-1]
    assert all('changed_source()' not in prompt for prompt in planner.calls)


def test_model_read_requests_and_automatic_review_share_two_reading_rounds(tmp_path):
    packet, helper, _ = imported_entrypoint_packet(tmp_path)
    extra = {'engine/core/first.py': 'FIRST_CONTEXT = 1\n',
             'engine/core/second.py': 'SECOND_CONTEXT = 2\n'}
    additional = packet_for(tmp_path, extra)
    packet['file_inventory'].update(additional['file_inventory'])

    class ReadingPlanner(UnsupportedPlanner):
        def chat(self, prompt, **kwargs):
            if len(self.calls) < 2:
                path = list(extra)[len(self.calls)]
                self.calls.append(prompt)
                return json.dumps({'request_files': [path]})
            return super().chat(prompt, **kwargs)

    planner = ReadingPlanner()
    with pytest.raises(PlanEvidenceError):
        propose_author_execution_plan(planner, packet, tmp_path)
    assert len(planner.calls) == 4
    assert all(path in packet['repository']['files'] for path in extra)
    assert helper not in packet['repository']['files']
    assert not packet['program_detected_author_commands'].get('source_review_reads')


@pytest.mark.parametrize('change', [
    lambda payload: payload.update(code='print("generated replacement")'),
    lambda payload: payload['repository'].update(revision='c' * 40),
    lambda payload: payload['steps'][0].update(argv=['python', '-c', 'print("inline replacement")']),
    lambda payload: payload['steps'][0].update(argv=['python', '../outside.py']),
])
def test_source_review_cannot_authorize_generated_unpinned_or_external_commands(tmp_path, change):
    packet, _, _ = imported_entrypoint_packet(tmp_path)
    invalid = execution_payload(packet)
    change(invalid)

    class UnsafePlanner(UnsupportedPlanner):
        def chat(self, prompt, **kwargs):
            if not self.calls:
                return super().chat(prompt, **kwargs)
            self.calls.append(prompt)
            return json.dumps(deepcopy(invalid))

    planner = UnsafePlanner()
    with pytest.raises(PlanEvidenceError) as captured:
        propose_author_execution_plan(planner, packet, tmp_path)
    details = captured.value.planning_evidence['failure_details']
    assert len(details) >= 2
    assert all(item['error'] for item in details)
    assert not (tmp_path / '__pycache__').exists()


def test_source_review_still_refuses_missing_required_data_without_invented_values(tmp_path):
    source = TRAIN.replace("parser.add_argument('--dataset', default='Cora')",
                           "parser.add_argument('--dataset', required=True)")
    packet = packet_for(tmp_path, {'README.md': 'Use the original training implementation.',
                                  'baseline.py': source})

    class IncompletePlanner(UnsupportedPlanner):
        def chat(self, prompt, **kwargs):
            if not self.calls:
                return super().chat(prompt, **kwargs)
            self.calls.append(prompt)
            return json.dumps(execution_payload(packet))

    planner = IncompletePlanner()
    with pytest.raises(PlanEvidenceError) as captured:
        propose_author_execution_plan(planner, packet, tmp_path)
    details = captured.value.planning_evidence['failure_details']
    argument_errors = [item['evidence'] for item in details
                       if (item.get('evidence') or {}).get('stage') == 'author_command_arguments']
    assert argument_errors
    assert all(item['missing_options'] == ['--dataset'] for item in argument_errors)
    assert all(item['argv'] == ['python', 'baseline.py'] for item in argument_errors)
    assert select_author_command(packet['program_detected_author_commands']) is None


@pytest.mark.parametrize('paths,text,expected', [
    (['a/train.py', 'b/train.py'], 'Use `train.py`.', set()),
    (['a/train.py', 'b/train.py'], 'Use `a/train.py`.', {'a/train.py'}),
    (['train.py', 'a/train.py'], 'Use `train.py`.', {'train.py'}),
    (['src/train.py'], 'Use `train.py`.', {'src/train.py'}),
    (['src/train.py'], 'Use `src/train.py`.', {'src/train.py'}),
    (['src/train.py'], 'Use src/train.py.bak instead.', set()),
    (['src/train.py'], 'Use src/train.py-extra instead.', set()),
    (['src/train.py'], 'Use src/train.py/child instead.', set()),
    (['src/train.py'], 'Use other/src/train.py instead.', set()),
    (['src/train.py'], 'Use train.py.', set()),
    (['src/train.py'], 'prefix_train.py postfix', set()),
    (['src/train.py'], '前train.py后', set()),
    (['src/train.py'], '前：train.py，后', {'src/train.py'}),
    (['src/train.py'], '(train.py), [train.py]; "train.py": <train.py>!', {'src/train.py'}),
    (['nested/fit(1)+a.py'], 'Use `fit(1)+a.py`.', {'nested/fit(1)+a.py'}),
    (['a', 'a+b.py', 'b.py'], 'a+b.py', {'a', 'a+b.py', 'b.py'}),
    (['a', 'a.py'], 'a.py', {'a.py'}),
    (['a.py', 'a.py+copy', 'copy'], 'a.py+copy', {'a.py', 'a.py+copy', 'copy'}),
    (['research/a+b.py', 'b.py'], 'research/a+b.py', {'research/a+b.py', 'b.py'}),
    (['README.md', 'docs/README.md'], 'See docs/README.md', {'docs/README.md'}),
    (['README.md', 'docs/README.md'], 'See README.md', {'README.md'}),
    (['a/train.py', 'b/train.py', 'run.py'], '', set()),
    ([], 'No source files are available.', set()),
])
def test_inventory_mention_matcher_preserves_path_ambiguity_boundaries_and_overlaps(paths, text, expected):
    from src.author_command_inference import _mentioned_paths, _PathMentionMatcher
    matcher = _PathMentionMatcher(paths)
    assert _mentioned_paths(text, paths) == expected
    assert matcher(text) == expected


def test_mention_matcher_cached_results_are_detached_and_inventory_scoped():
    from src.author_command_inference import _PathMentionMatcher
    unique = _PathMentionMatcher(['a/train.py'])
    first = unique('Use `train.py`.')
    first.clear()
    assert unique('Use `train.py`.') == {'a/train.py'}
    first.add('invented.py')
    assert unique('Use `train.py`.') == {'a/train.py'}
    ambiguous = _PathMentionMatcher(['a/train.py', 'b/train.py'])
    assert ambiguous('Use `train.py`.') == set()
    assert unique('Use `train.py`.') == {'a/train.py'}


def test_discovery_reuses_one_mention_index_but_rebuilds_it_for_next_inventory(tmp_path, monkeypatch):
    from src import author_command_inference as inference
    original = inference._PathMentionMatcher
    built, scans = [], []

    class ObservedMatcher:
        def __init__(self, paths):
            built.append(frozenset(paths))
            self.matcher = original(paths)

        def __call__(self, text):
            scans.append(text)
            return self.matcher(text)

    monkeypatch.setattr(inference, '_PathMentionMatcher', ObservedMatcher)
    first = packet_for(tmp_path / 'first', {
        'README.md': 'Training uses `baseline.py`.\nRun the training from the repository root.',
        'research/baseline.py': 'import learner\nlearner.fit(records)\n',
    })
    first_discovery = discover_author_commands(first, tmp_path / 'first')
    assert len(built) == 1 and len(scans) > 1
    first_candidate = select_author_command(first_discovery)
    assert first_candidate['source_path'] == 'research/baseline.py'
    assert first_candidate['readme_evidence']

    second = packet_for(tmp_path / 'second', {
        'README.md': 'Training uses `baseline.py`.',
        'research/baseline.py': 'import learner\nlearner.fit(records)\n',
        'alternative/baseline.py': 'import learner\nlearner.fit(records)\n',
    })
    second_discovery = discover_author_commands(second, tmp_path / 'second')
    assert len(built) == 2 and built[0] != built[1]
    assert select_author_command(second_discovery) is None
    assert all(not candidate['readme_evidence'] for candidate in second_discovery['candidates'])

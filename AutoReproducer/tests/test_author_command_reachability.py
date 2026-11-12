"""Deferred or unreachable training calls never authorize a source-only fallback."""
import ast

import pytest

from src.author_command_inference import (
    _training_evidence, discover_author_commands, select_author_command,
)
from src.author_execution_plan import propose_author_execution_plan
from src.discovered_repository_plan import PlanEvidenceError
from test_author_command_inference import packet_for


@pytest.mark.parametrize('source', [
    "runner = lambda: estimator.fit(records)\nprint('configured only')\n",
    "steps = (estimator.fit(row) for row in records)\nprint('configured only')\n",
    "def run_experiment():\n    yield estimator.fit(records)\nrun_experiment()\n",
    "def train():\n    yield estimator.fit(records)\ntrain()\n",
    "def train():\n    yield from records\n    estimator.fit(records)\ntrain()\n",
    "async def run_experiment():\n    estimator.fit(records)\nrun_experiment()\n",
    "async def train():\n    estimator.fit(records)\ntrain()\n",
    "def main():\n    return\n    estimator.fit(records)\nmain()\n",
    "raise RuntimeError('stop')\nestimator.fit(records)\n",
    "for epoch in (1,):\n    break\n    estimator.fit(records)\n",
    "for epoch in (1,):\n    continue\n    estimator.fit(records)\n",
    "for row in []:\n    estimator.fit(row)\n",
    "for row in ():\n    estimator.fit(row)\n",
    "steps = [estimator.fit(row) for row in []]\n",
    "steps = {estimator.fit(row) for row in ()}\n",
    "steps = {row: estimator.fit(row) for row in {}}\n",
    "if unresolved_mode:\n    estimator.fit(records)\n",
])
def test_deferred_and_unreachable_calls_cannot_select_an_author_command(tmp_path, source):
    packet = packet_for(tmp_path, {
        'README.md': 'The original experiment entrypoint is train.py.',
        'train.py': source,
    })
    discovery = discover_author_commands(packet, tmp_path)
    assert discovery['candidates'] == []
    assert select_author_command(discovery) is None
    assert all(not row['training_evidence'] for row in discovery['review_candidates'])
    with pytest.raises(PlanEvidenceError, match='No unique complete source-derived'):
        propose_author_execution_plan(None, packet, tmp_path)
    assert not list(tmp_path.rglob('*.pyc'))


@pytest.mark.parametrize('source', [
    'estimator.fit(records)\n',
    'def main():\n    estimator.fit(records)\nmain()\n',
    "if __name__ == '__main__':\n    estimator.fit(records)\n",
    'for row in (1,):\n    estimator.fit(row)\n',
    'for row in []:\n    pass\nelse:\n    estimator.fit(records)\n',
    'runner = lambda value=estimator.fit(records): value\n',
    'steps = (row for row in estimator.fit(records))\n',
    'def main():\n    return estimator.fit(records)\nmain()\n',
    "def main():\n    global current\n    estimator.fit(records)\nmain()\n",
    'def main():\n    def rows():\n        yield 1\n    estimator.fit(records)\nmain()\n',
])
def test_reachable_training_and_eager_expressions_remain_navigation_evidence(source):
    evidence = _training_evidence(ast.parse(source), source)
    assert len(evidence) == 1
    assert 'estimator.fit(' in evidence[0]['quote']

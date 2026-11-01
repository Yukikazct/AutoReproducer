"""Frozen summary values survive uncertain units; structured output stays authoritative."""
import hashlib

import pytest

from src import author_unlabelled_metrics as summaries
from src.author_unlabelled_metrics import numeric_summaries, validate_summary_labels
from test_author_unlabelled_metrics import LOGGER_OUTPUT, LOGGER_SOURCE, logger_summary
from test_author_summary_supplement import collect_author_metrics, compare, completed


@pytest.mark.parametrize('mutation', ['result /= 100', 'result = torch.tensor(self.results)',
                                    'result[0] = 0', 'result.mul_(.01)', 'mutate(result)'])
def test_same_line_modification_cannot_prove_summary_percentage(logger_summary, mutation):
    payload, candidates, packet, root = logger_summary
    text = LOGGER_SOURCE.replace('result = 100 * torch.tensor(self.results)',
                                'result = 100 * torch.tensor(self.results); ' + mutation)
    (root / 'logger.py').write_text(text, encoding='utf-8')
    sha = hashlib.sha256((root / 'logger.py').read_bytes()).hexdigest()
    packet['repository']['files']['logger.py'] = {'text': text, 'sha256': sha}
    payload['citations'][0].update(quote=text, sha256=sha)
    row = validate_summary_labels(payload, candidates, packet, root)[0]
    assert row['unit'] == '' and row['value'] == 57.9 and row['std'] == 8.41


def test_same_line_identical_expression_needs_the_quoted_source_span(logger_summary):
    payload, candidates, packet, root = logger_summary
    text = LOGGER_SOURCE.replace('result = 100 * torch.tensor(self.results)',
        'unrelated = 100 * torch.tensor(self.results); result = 100 * torch.tensor(self.results)')
    (root / 'logger.py').write_text(text, encoding='utf-8')
    sha = hashlib.sha256((root / 'logger.py').read_bytes()).hexdigest()
    packet['repository']['files']['logger.py'] = {'text': text, 'sha256': sha}
    payload['citations'][0].update(quote=text, sha256=sha)
    payload['metrics'][0]['unit_quote'] = 'unrelated = 100 * torch.tensor(self.results)'
    row = validate_summary_labels(payload, candidates, packet, root)[0]
    assert row['unit'] == '' and row['value'] == 57.9


@pytest.mark.parametrize('marker_stream', ['stdout', 'stderr'])
@pytest.mark.parametrize('payload', ['[]', '{"accuracy": 0.8}', '{broken'])
def test_marker_on_either_stream_blocks_numeric_fallback_for_the_whole_step(marker_stream, payload):
    other = 'stderr' if marker_stream == 'stdout' else 'stdout'
    execution = completed('')
    execution['steps'][0][marker_stream] = 'AUTOREPRO_METRICS:' + payload
    execution['steps'][0][other] = LOGGER_OUTPUT
    assert numeric_summaries(execution) == []


@pytest.mark.parametrize('marker_stream', ['stdout', 'stderr'])
@pytest.mark.parametrize('payload', ['[]', '{"accuracy": 0.8}', '{broken'])
def test_comparison_hook_independently_preserves_cross_stream_marker_authority(
        tmp_path, monkeypatch, marker_stream, payload):
    other = 'stderr' if marker_stream == 'stdout' else 'stdout'
    execution = completed('')
    execution['steps'][0][marker_stream] = 'AUTOREPRO_METRICS:' + payload
    execution['steps'][0][other] = 'Loss:0\n0.9 0.1'
    original, _ = collect_author_metrics(execution)
    monkeypatch.setattr(summaries, 'numeric_summaries', lambda saved: [{
        'id': 'train:' + other + ':2', 'step_id': 'train', 'stream': other,
        'source_quote': '0.9 0.1', 'values': [0.9, 0.1]}])

    def interpret(*args, **kwargs):
        pytest.fail('Either-stream authoritative marker must block numeric supplementation')

    result, _ = compare(tmp_path, monkeypatch, execution, interpret)
    assert result['metric_records'] == original

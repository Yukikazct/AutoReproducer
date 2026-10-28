"""Exercise ordinary AI verdicts through the production repository service."""
import json
from pathlib import Path

import pytest

from src.method_adapters import read_json
from test_author_claim_comparison_service import ComparisonLLM, comparison_service


class AssessmentLLM(ComparisonLLM):
    def __init__(self, fixture, *, verdict='reproduced', comparison_outcome='matched',
                 assessment_error=None, diagnosis_error=None):
        super().__init__(fixture, comparison_outcome)
        self.verdict = verdict
        self.assessment_error = assessment_error
        self.diagnosis_error = diagnosis_error

    def chat(self, prompt, **kwargs):
        if kwargs['task'] == 'result_diagnosis' and self.diagnosis_error:
            raise self.diagnosis_error
        if kwargs['task'] != 'reproduction_assessment':
            return super().chat(prompt, **kwargs)
        self.calls.append((prompt, kwargs))
        if self.assessment_error:
            raise self.assessment_error
        evidence = json.loads(prompt.split('已验证证据：\n', 1)[1])
        return json.dumps({'verdict': self.verdict, 'confidence': 0.84,
            'reason': '按实测结果和实验差异评估本次实验。', 'limitations': ['本测试使用构造的论文和指标。'],
            'metric_assessments': [{'actual_id': row['actual']['id'], 'verdict': self.verdict,
                'reason': '测试模型选择适用于当前实验的误差范围。',
                'tolerance': {'absolute': 7.0, 'relative': None,
                              'reason': '模拟模型根据实验设置选择 7 个百分点范围，用于集成接线测试。'}}
                for row in evidence['metrics']]})


@pytest.mark.parametrize('verdict,flag', [
    ('reproduced', True), ('partially_reproduced', None),
    ('not_reproduced', False), ('insufficient_evidence', None)])
def test_pdf_service_applies_and_persists_ai_judgement(tmp_path, monkeypatch, verdict, flag):
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    runner.llm = AssessmentLLM(fixture, verdict=verdict)
    events = []
    result = runner.run(request, on_event=events.append)
    assert result['state'] == 'COMPLETED', result.get('error')
    validation = result['data']['validation']
    assert validation['status'] == 'ai_' + verdict
    assert validation['is_reproduced'] is flag
    assert validation['paper_metrics_verified'] is False
    assert validation['execution_status'] == 'author_code_completed'
    assessment = validation['reproduction_assessment']
    assert assessment['source'] == 'ai' and assessment['status'] == 'completed'
    comparison = validation['author_claim_comparison']['comparisons'][0]
    assert comparison['difference'] == pytest.approx(-6.0)
    assert comparison['meets_reference'] is False  # Old 5% reference is no longer the verdict.
    assert assessment['metric_assessments'][0]['actual'] == comparison['actual']
    root = Path(result['data']['run_dir'])
    assert read_json(root / 'reproduction_assessment.json') == assessment
    assert read_json(root / 'author_execution_verification.json') == validation
    assert read_json(root / 'execution.json')['author_execution_verification'] == validation
    assert read_json(root / 'result.json')['data']['validation'] == validation
    assert 'AI 复现判断' in result['data']['report']
    assert assessment['reason'] in result['data']['report']
    stages = [event['phase_id'] for event in events
              if event.get('type') == 'state' and event['status'] == 'running']
    assert stages.index('compare_author_claims') < stages.index('assess_reproduction') < stages.index('generate_report')
    if verdict == 'reproduced':
        assert all(kwargs['task'] != 'result_diagnosis' for _, kwargs in runner.llm.calls)
        assert '未达到参考' not in result['data']['report']


def test_reference_only_settings_do_not_prevent_ai_success(tmp_path, monkeypatch):
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    runner.llm = AssessmentLLM(fixture, comparison_outcome='unknown')
    result = runner.run(request)
    validation = result['data']['validation']
    assert validation['author_claim_comparison']['comparisons'][0]['status'] == 'reference_only'
    assert validation['status'] == 'ai_reproduced' and validation['is_reproduced'] is True


def test_api_failure_preserves_successful_execution(tmp_path, monkeypatch):
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    runner.llm = AssessmentLLM(fixture, assessment_error=RuntimeError('provider unavailable'))
    result = runner.run(request)
    assert result['state'] == 'COMPLETED'
    validation = result['data']['validation']
    assert validation['status'] == 'author_code_completed' and validation['is_reproduced'] is None
    assert validation['reproduction_assessment']['status'] == 'unavailable'
    assert validation['reproduction_assessment']['reason'] in result['data']['report']


def test_cancelling_ai_assessment_preserves_execution_and_comparison(tmp_path, monkeypatch):
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    runner.llm = AssessmentLLM(fixture, assessment_error=KeyboardInterrupt('assessment cancelled'))
    with pytest.raises(KeyboardInterrupt, match='assessment cancelled'):
        runner.run(request)
    result = read_json(next((runner.root / 'runs').iterdir()) / 'result.json')
    validation = result['data']['validation']
    assert validation['status'] == 'author_code_completed'
    assert validation['author_claim_comparison']['status'] == 'compared'
    assert validation['reproduction_assessment']['interrupted'] is True
    assert result['data']['execution']['success'] is True


def test_cancelling_later_diagnosis_preserves_completed_ai_judgement(tmp_path, monkeypatch):
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    runner.llm = AssessmentLLM(fixture, verdict='not_reproduced',
                              diagnosis_error=KeyboardInterrupt('diagnosis cancelled'))
    with pytest.raises(KeyboardInterrupt, match='diagnosis cancelled'):
        runner.run(request)
    result = read_json(next((runner.root / 'runs').iterdir()) / 'result.json')
    validation = result['data']['validation']
    assert validation['status'] == 'ai_not_reproduced'
    assert validation['reproduction_assessment']['status'] == 'completed'
    assert validation['author_claim_comparison']['status'] == 'compared'
    assert result['data']['execution']['success'] is True


def test_optional_assessment_file_error_does_not_erase_verdict(tmp_path, monkeypatch):
    from src import discovered_repository_reproduction as service
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    runner.llm = AssessmentLLM(fixture)
    persist = service.write_json

    def failing_write(path, value):
        if Path(path).name == 'reproduction_assessment.json':
            raise OSError('read-only destination')
        return persist(path, value)

    monkeypatch.setattr(service, 'write_json', failing_write)
    result = runner.run(request)
    validation = result['data']['validation']
    assert result['state'] == 'COMPLETED' and validation['status'] == 'ai_reproduced'
    assert any('reproduction_assessment.json' in value for value in validation['warnings'])
    assert read_json(Path(result['data']['run_dir']) / 'result.json')['data']['validation'] == validation

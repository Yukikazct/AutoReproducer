"""A PDF's framework link cannot suppress discovery of its implementation."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
from copy import deepcopy
import shutil

from discovered_fixtures import make_experiment, write_pdf
from src import discovered_repository_reproduction as service
from src.agents.repository_verifier import RepositoryVerifierAgent
from src.audit.audit_logger import AuditLogger
from src import author_execution_plan

from src.agents import resource_finder
from src.discovered_repository_reproduction import repository_candidates


def test_framework_link_does_not_bypass_title_search_or_lock_model(monkeypatch):
    framework = 'https://github.com/dmlc/dgl'
    implementation = 'https://github.com/researchers/graph-method'
    search = Mock(return_value={'selected_repo': implementation,
        'candidates': [{'repo_urls': [implementation], 'source': 'github_search'}],
        'discovery_chain': ['github_search']})
    monkeypatch.setattr(resource_finder, 'discover_repositories', search)
    llm = Mock()
    llm.chat.return_value = json.dumps({'code_repo_url': implementation, 'alternative_repos': []})
    llm.get_call_count.return_value = 1
    link = {'url': framework, 'source': 'pdf_text', 'page': 9,
            'evidence_type': 'repository_link', 'is_author_code': False}
    resources = resource_finder.ResourceFinderAgent(llm).run({
        'paper_info': {'title': 'An optimization framework for graph networks', 'code_url': framework},
        'extracted_repository_links': [link]})['resources']
    assert search.call_args.kwargs['preferred_url'] == ''
    assert resources['code_repo_url'] == implementation
    assert resources['repo_discovery']['selected_repo'] == implementation
    assert framework in resources['alternative_repos']
    assert resources['repository_identity']['status'] == 'candidate_unverified'


def test_discovered_alternate_is_attempted_before_other_pdf_frameworks():
    first = {'url': 'https://github.com/framework/a', 'source': 'pdf_text',
             'evidence_type': 'repository_link', 'is_author_code': False}
    other = {**first, 'url': 'https://github.com/framework/b'}
    implementation = 'https://github.com/researchers/method'
    doc = SimpleNamespace(repository_links=[first, other])
    candidates = repository_candidates(doc, first, {'alternative_repos': [implementation, implementation]},
                                       execution_only=True)
    assert [item['url'] for item in candidates] == [first['url'], implementation, other['url']]
    assert candidates[1]['source'] == 'model_discovered'
    assert candidates[1]['is_author_code'] is False
    assert [item['url'] for item in repository_candidates(doc, first, {'alternative_repos': [implementation]})] == [
        first['url'], implementation, other['url']]


def test_manual_repository_is_not_replaced_by_discovery():
    evidence = {'url': 'https://github.com/user/method', 'source': 'user_supplied'}
    assert repository_candidates(SimpleNamespace(repository_links=[]), evidence,
        {'alternative_repos': ['https://github.com/other/method']}, execution_only=True) == [evidence]


def test_service_reads_discovered_alternate_after_download_failure_without_review(tmp_path, monkeypatch):
    fixture = make_experiment(tmp_path)
    framework = 'https://github.com/framework/tool'
    implementation = fixture['snapshot']['url']
    write_pdf(fixture['pdf'], [['Separating Signed Signals', 'Abstract',
                              'We use a framework: ' + framework]])
    exported = []
    def export(root, profile, workspace, **kwargs):
        exported.append(profile['repository']['url'])
        if exported[-1] == framework:
            raise OSError('Primary candidate download failed')
        shutil.copytree(fixture['workspace'], workspace)
        return {**deepcopy(fixture['snapshot']), **profile['repository'],
                'resolved_sha': profile['repository']['revision']}
    review = Mock(side_effect=AssertionError('Execution mode must not call the identity reviewer'))
    monkeypatch.setattr(service, 'export_repository', export)
    monkeypatch.setattr(service, 'resolve_repository_revision', lambda *a, **kw: ('a' * 40, {}))
    monkeypatch.setattr(RepositoryVerifierAgent, 'run', review)
    stop = Mock(side_effect=ValueError('Reached matching implementation planning'))
    monkeypatch.setattr(author_execution_plan, 'propose_author_execution_plan', stop)
    runner = service.DiscoveredRepositoryReproduction(tmp_path / 'service',
        AuditLogger(log_dir=str(tmp_path / 'logs'), ledger_dir=str(tmp_path / 'ledger')), llm=Mock())
    runner.llm.get_call_count.return_value = 0
    result = runner.run({'pdf_path': str(fixture['pdf']), 'author_execution_only': True,
        'resources': {'code_repo_url': framework, 'alternative_repos': [implementation]}})
    assert exported == [framework, implementation]
    review.assert_not_called()
    attempts = result['data']['repository_identity_attempts']
    assert not attempts[0]['accepted'] and attempts[1]['accepted']
    assert attempts[0]['error'] == 'Primary candidate download failed'
    assert result['data']['resources']['selected_repo'] == implementation
    assert result['data']['validation']['failure_phase'] == 'plan_repository'
    stop.assert_called_once()

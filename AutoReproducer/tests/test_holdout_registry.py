"""Eligibility is bound to paper identity, actual PDF bytes and the frozen product."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import holdout_registry as registry
from scripts import verify_browser_pdf_holdout as browser


@pytest.fixture
def paper_root(tmp_path, monkeypatch):
    for name in ('app.py', 'src/module.py', 'frontend/page.py', 'requirements.txt',
                 'frontend/styles.css', 'scripts/freeze_holdout_source.py', 'scripts/verify_browser_pdf_holdout.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture\n', encoding='utf-8')
    paper = {'id': 'fixture-paper', 'title': 'Fixture Paper', 'aliases': ['FP'],
             'source_ids': ['arxiv:2601.12345v1'], 'pdf_sha256': [hashlib.sha256(b'pdf').hexdigest()],
             'repository_url': 'https://github.com/fixture/author', 'status': 'candidate',
             'repository_revision': '1' * 40,
             'history': [{'outcome': 'registered_unused_candidate', 'evidence': 'fixture acquisition receipt'}]}
    data = {'version': 1, 'papers': [paper]}
    path = tmp_path / registry.REGISTRY
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(registry.subprocess, 'check_output', lambda *a, **k: 'a' * 40)
    return tmp_path, data


def store(root, data):
    (root / registry.REGISTRY).write_text(json.dumps(data), encoding='utf-8')


def verify(root, freeze, **kwargs):
    args = dict(paper_id='FP', pdf_sha256=hashlib.sha256(b'pdf').hexdigest(),
                repository='https://github.com/FIXTURE/author.git', require_independent=True)
    args.update(kwargs)
    return registry.verify_freeze(root, freeze, **args)


def test_unused_registered_pdf_and_alias_match_the_same_frozen_paper(paper_root):
    root, _ = paper_root
    freeze = registry.build_freeze(root, paper_id='Fixture Paper')
    assert verify(root, freeze)['paper_id'] == 'fixture-paper'
    assert 'requirements.txt' in freeze['files']
    assert registry.write_freeze(root, 'fixture', freeze).is_file()
    with pytest.raises(FileExistsError):
        registry.write_freeze(root, 'fixture', freeze)
    with pytest.raises(ValueError):
        registry.write_freeze(root, '../escape', freeze)


@pytest.mark.parametrize('field,value', [('aliases', ['F-P']), ('pdf_sha256', [hashlib.sha256(b'pdf').hexdigest()]),
                                        ('source_ids', ['https://arxiv.org/pdf/2601.12345v7.pdf'])])
def test_duplicate_title_alias_pdf_or_arxiv_version_cannot_create_new_identity(paper_root, field, value):
    root, data = paper_root
    other = deepcopy(data['papers'][0])
    other.update(id='different', title='Different', aliases=['Other'], source_ids=['arxiv:2601.54321'],
                 pdf_sha256=['b' * 64])
    other[field] = value
    data['papers'].append(other)
    store(root, data)
    with pytest.raises(ValueError, match='Conflicting'):
        registry.load_registry(root)


def test_retired_paper_alias_is_rejected_and_changed_registry_invalidates_freeze(paper_root):
    root, data = paper_root
    freeze = registry.build_freeze(root, paper_id='FP')
    data['papers'][0].update(status='regression', retirement_reason='Product repaired after the trial')
    store(root, data)
    with pytest.raises(ValueError, match='not an unused'):
        registry.build_freeze(root, paper_id='F-P')
    with pytest.raises(ValueError):
        verify(root, freeze)


def test_status_cannot_contradict_historical_trial_or_unknown_state(paper_root):
    root, data = paper_root
    data['papers'][0]['history'].append({'outcome': 'retired_after_repair', 'evidence': 'old trial'})
    store(root, data)
    with pytest.raises(ValueError, match='conflicts'):
        registry.load_registry(root)
    data['papers'][0]['status'] = 'unknown'
    store(root, data)
    with pytest.raises(ValueError, match='Unknown'):
        registry.load_registry(root)


@pytest.mark.parametrize('change', ['source', 'added_source', 'removed_source', 'frontend_asset', 'requirements', 'registry'])
def test_changed_source_file_set_dependencies_or_registry_reject_completion(paper_root, change):
    root, data = paper_root
    freeze = registry.build_freeze(root, paper_id='FP')
    if change == 'source':
        (root / 'src/module.py').write_text('# changed')
    elif change == 'added_source':
        (root / 'src/new.py').write_text('# new')
    elif change == 'removed_source':
        (root / 'src/module.py').unlink()
    elif change == 'requirements':
        (root / 'requirements.txt').write_text('changed==1')
    elif change == 'frontend_asset':
        (root / 'frontend/styles.css').write_text('body { color: red; }')
    else:
        data['papers'][0]['aliases'].append('new alias')
        store(root, data)
    with pytest.raises(ValueError, match='changed'):
        verify(root, freeze)


def test_unregistered_pdf_or_repository_and_unknown_identity_are_rejected(paper_root):
    root, _ = paper_root
    freeze = registry.build_freeze(root, paper_id='FP')
    for kwargs in ({'pdf_sha256': '0' * 64}, {'repository': 'https://github.com/fixture/fork'},
                   {'paper_id': 'unknown'}):
        with pytest.raises(ValueError):
            verify(root, freeze, **kwargs)


def test_legacy_freeze_is_regression_only_and_independent_requires_comparison(paper_root, monkeypatch):
    root, _ = paper_root
    freeze = {'files': {'app.py': hashlib.sha256((root / 'app.py').read_bytes()).hexdigest()}}
    assert registry.verify_freeze(root, freeze)['independent'] is False
    with pytest.raises(ValueError, match='version-2'):
        verify(root, freeze)
    monkeypatch.setattr(browser, 'ROOT', root)
    args = SimpleNamespace(require_independent=True, require_comparison=False, paper_id='FP',
                           repo='https://github.com/fixture/author')
    with pytest.raises(ValueError, match='require-comparison'):
        browser.verify_acceptance_source(args, freeze, '0' * 64)


def test_independent_freeze_requires_a_pinned_author_commit(paper_root):
    root, data = paper_root
    del data['papers'][0]['repository_revision']
    store(root, data)
    with pytest.raises(ValueError, match='pinned'):
        registry.build_freeze(root, paper_id='FP')


def test_independent_completion_rejects_changed_installed_dependencies(paper_root, monkeypatch):
    root, _ = paper_root
    freeze = registry.build_freeze(root, paper_id='FP')
    assert freeze['environment']['distributions']
    monkeypatch.setattr(registry, 'runtime_dependencies', lambda: {'distributions': {'changed': '2'}})
    with pytest.raises(ValueError, match='environment'):
        verify(root, freeze)


def test_independent_completion_rejects_different_git_commit(paper_root, monkeypatch):
    root, _ = paper_root
    freeze = registry.build_freeze(root, paper_id='FP')
    monkeypatch.setattr(registry.subprocess, 'check_output', lambda *a, **k: 'b' * 40)
    with pytest.raises(ValueError, match='Git commit'):
        verify(root, freeze)


def test_json_round_trip_preserves_freeze_and_dependency_evidence(paper_root):
    root, _ = paper_root
    freeze = json.loads(json.dumps(registry.build_freeze(root, paper_id='FP')))
    assert verify(root, freeze)['independent'] is True


def test_project_registry_contains_retired_recent_papers_and_all_archived_pdf_hashes():
    root = Path(__file__).resolve().parents[1]
    data = registry.load_registry(root)
    for alias in ('ccassg', 'difformer', 'nodeformer', 'rfcn', 'correctandsmooth', 'graphcon'):
        assert registry.select_paper(data, alias)['status'] == 'regression'
    recorded = {sha for paper in data['papers'] for sha in paper['pdf_sha256']}
    for name in ('examples/papers/manifest.json', 'examples/papers/candidates_20261012/manifest.json',
                 'examples/papers/candidates_20261017/manifest.json'):
        manifest = json.loads((root / name).read_text('utf-8'))
        assert all(p['sha256'] in recorded for p in manifest['papers'])


def test_historical_runtime_receipts_and_wrong_paper_download_keep_separate_identities():
    root = Path(__file__).resolve().parents[1]
    data = registry.load_registry(root)
    receipts = json.loads((root / 'docs/evidence/historical_pdf_provenance_20261017.json').read_text('utf-8'))
    for receipt in receipts['receipts']:
        paper = registry.select_paper(data, receipt['paper_id'])
        assert paper['status'] == 'regression'
        assert receipt['pdf_sha256'] in paper['pdf_sha256']
    rejected = next(row for row in receipts['receipts'] if row['original_preflight_rejected'])
    assert rejected['pdf_sha256'] not in registry.select_paper(data, 'nodeformer')['pdf_sha256']
    assert registry.select_paper(data, rejected['paper_id'])['title'].startswith('Prediction of quaternary hydrides')

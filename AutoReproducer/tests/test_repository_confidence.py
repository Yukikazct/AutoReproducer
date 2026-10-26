"""Missing confidence is not zero; recorded linkage is not a new verdict."""
from copy import deepcopy

import pytest

from src.repository_confidence import repository_confidence_lines


def direct_record():
    url = 'https://github.com/ivam-he/ChebNetII'
    return {'resources': {'code_repo_url': url,
        'repository_identity': {'source': 'pinned_git_direct_execution', 'author_identity_confirmed': False},
        'selection_evidence': {'url': url, 'raw_url': url, 'page': 1, 'source': 'pdf_text',
            'is_author_code': True, 'evidence_type': 'author_code_statement',
            'context': 'Our code is available at ' + url}},
        'pdf_input': {'sha256': 'f' * 64, 'pages': 20},
        'repository': {'url': url, 'resolved_sha': 'a' * 40}}


def test_direct_author_statement_has_evidence_without_invented_confidence():
    data = direct_record()
    original = deepcopy(data)
    text = '\n'.join(repository_confidence_lines(data))
    assert '未评分' in text and '0.00' not in text
    assert 'PDF 第 1 页的作者代码声明链接与本次选择仓库一致' in text
    assert '未记录独立的论文与实现匹配审核结果' in text
    assert data == original


@pytest.mark.parametrize('score,expected', [(0, '0.00'), (1, '1.00'), ('0.87', '0.87')])
def test_explicit_scores_including_zero_are_retained(score, expected):
    data = direct_record()
    data['resources']['confidence'] = score
    assert expected in repository_confidence_lines(data)[0]
    assert '未经概率校准' in repository_confidence_lines(data)[0]


@pytest.mark.parametrize('score', [None, '', 'invalid', True, float('nan'), float('inf'), -1, 1.2])
def test_missing_or_invalid_scores_are_unscored(score):
    data = direct_record()
    data['resources']['confidence'] = score
    assert '未评分' in repository_confidence_lines(data)[0]


@pytest.mark.parametrize('mutation', ['different_repository', 'different_context', 'missing_pdf', 'out_of_range_page'])
def test_unbound_statements_cannot_be_presented_as_matching_author_evidence(mutation):
    data = direct_record()
    if mutation == 'different_repository':
        data['resources']['selection_evidence']['url'] = 'https://github.com/other/repository'
    elif mutation == 'different_context':
        data['resources']['selection_evidence']['context'] = 'An unrelated citation.'
    elif mutation == 'missing_pdf':
        data['pdf_input'].clear()
    else:
        data['resources']['selection_evidence']['page'] = 21
    assert '的作者代码声明链接与本次选择仓库一致' not in '\n'.join(repository_confidence_lines(data))


def test_review_must_refer_to_current_fixed_revision():
    data = direct_record()
    review = {'accepted': True, 'repository_revision': 'b' * 40}
    data['resources']['repository_identity']['review'] = review
    assert '未绑定当前固定版本' in repository_confidence_lines(data)[-1]
    review['repository_revision'] = 'a' * 40
    assert '审核版本与本次固定版本一致' in repository_confidence_lines(data)[-1]


def test_github_case_and_malformed_historical_urls_are_handled():
    data = direct_record()
    data['resources']['code_repo_url'] = data['resources']['code_repo_url'].lower()
    assert '的作者代码声明链接与本次选择仓库一致' in '\n'.join(repository_confidence_lines(data))
    data['resources']['code_repo_url'] = {'invalid': 'url'}
    assert '尚无与本次仓库对应' in '\n'.join(repository_confidence_lines(data))


def test_candidate_association_limit_remains_in_readable_explanation():
    data = direct_record()
    data['resources']['repository_identity'] = {
        'source': 'github_search', 'reason': '候选，尚未核验与论文作者关联。'}
    assert '候选，尚未核验与论文作者关联' in '\n'.join(repository_confidence_lines(data))

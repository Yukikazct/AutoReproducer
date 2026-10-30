"""Describe repository association evidence without inventing a probability."""
import math
import re

from src.agents.repo_discovery import normalize_github_repo_url


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _repository(value):
    return normalize_github_repo_url(value).casefold() if isinstance(value, str) else ''


def _score(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def repository_confidence_lines(data):
    """Keep missing scores, PDF linkage, version pins and semantic review distinct.

    A direct PDF author statement is source evidence, not a calibrated model
    probability or an independent implementation review. This is presentation
    only: the original score and acceptance fields are never modified.
    """
    resources = _mapping(data.get('resources'))
    identity = _mapping(resources.get('repository_identity'))
    repository = _mapping(data.get('repository'))
    evidence = _mapping(resources.get('selection_evidence')) or _mapping(identity.get('evidence'))
    pdf = _mapping(data.get('pdf_input')) or _mapping(data.get('pdf_resolution'))
    score = _score(resources.get('confidence'))
    if score is None:
        explanation = ('固定仓库直接执行路径未提供匹配评分'
                       if identity.get('source') == 'pinned_git_direct_execution' else '未记录有效匹配评分')
        lines = ['- **仓库匹配置信度**: 未评分（' + explanation + '）。']
    else:
        lines = [f'- **仓库匹配置信度**: {score:.2f}（已记录的资源发现评分，未经概率校准）。']

    selected = _repository(resources.get('code_repo_url') or resources.get('selected_repo'))
    linked = _repository(evidence.get('url'))
    page = evidence.get('page')
    context = evidence.get('context')
    source_url = evidence.get('raw_url') or evidence.get('url')
    bound_statement = (
        evidence.get('is_author_code') is True
        and evidence.get('evidence_type') == 'author_code_statement'
        and evidence.get('source') == 'pdf_text'
        and type(page) is int and page > 0
        and (type(pdf.get('pages')) is not int or page <= pdf['pages'])
        and isinstance(pdf.get('sha256'), str) and re.fullmatch(r'[0-9a-fA-F]{64}', pdf['sha256'])
        and isinstance(context, str) and isinstance(source_url, str) and source_url in context
        and selected and linked == selected
    )
    if bound_statement:
        lines.append(f'- **仓库关联依据**: PDF 第 {page} 页的作者代码声明链接与本次选择仓库一致；来源 PDF 已记录摘要。')
    elif selected and linked == selected and type(page) is int and page > 0:
        lines.append(f'- **仓库关联依据**: 已记录 PDF 第 {page} 页的同仓库链接；作者声明或来源绑定证据尚不完整。')
    else:
        lines.append('- **仓库关联依据**: 尚无与本次仓库对应的完整 PDF 作者声明记录。')

    reason = identity.get('reason')
    if isinstance(reason, str) and reason.strip() and identity.get('source') != 'pinned_git_direct_execution':
        summary = ' '.join(reason.split())
        if len(summary) > 300:
            summary = summary[:300] + '（完整说明见详情）'
        summary = re.sub(r'([\\`*_{}\[\]()<>!|])', r'\\\1', summary)
        lines.append('- **关联核验说明**: ' + summary)

    revision = repository.get('resolved_sha')
    fetched = _repository(repository.get('url'))
    if (selected and fetched == selected and isinstance(revision, str)
            and re.fullmatch(r'[0-9a-fA-F]{40}', revision)):
        lines.append('- **源码固定**: 已记录所选仓库的固定 Git 版本；固定版本本身不证明实现与论文一致。')
    review = _mapping(identity.get('review'))
    if review.get('accepted') is True:
        if revision and review.get('repository_revision') == revision:
            lines.append('- **实现匹配审核**: 已保存的审核通过，审核版本与本次固定版本一致。')
        else:
            lines.append('- **实现匹配审核**: 已保存审核未绑定当前固定版本，当前版本的匹配仍待核验。')
    else:
        lines.append('- **实现匹配审核**: 未记录独立的论文与实现匹配审核结果。')
    return lines

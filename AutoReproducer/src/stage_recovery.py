"""Evidence-led recovery before an author repository has been prepared.

Only the existing reader/finder are callable here. PDF bytes, URI annotations
and explicit user choices remain authoritative; model feedback cannot select a
different experiment or manufacture missing paper content.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path

from src.intelligent_recovery import current_investigation, InvestigationStopped


PHASES = {'READ_PAPER': 'paper_parsing', 'FIND_RESOURCES': 'repository_discovery',
          'BUILD_ENV': 'environment_preparation', 'EXECUTE_CODE': 'author_execution'}


class StageEvidenceError(ValueError):
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def _check_original_pdf(data, result=None):
    expected = (data.get('pdf_input') or {}).get('sha256')
    if not expected or not data.get('pdf_path'):
        return
    observed = ((result or {}).get('pdf_input') or {}).get('sha256')
    if (observed and observed != expected
            or hashlib.sha256(Path(data['pdf_path']).read_bytes()).hexdigest() != expected):
        current_investigation().stop('source_changed', requires_human=True)
        raise InvestigationStopped('Original PDF changed during recovery')


def _verify(stage, result, data):
    if not isinstance(result, dict):
        return {'success': False, 'reason': 'Stage did not return structured evidence'}
    if stage == 'READ_PAPER':
        coverage = result.get('reading_coverage') or {}
        complete = coverage.get('complete') is True
        identity = bool((result.get('paper_info') or {}).get('title'))
        return {'success': complete and identity,
                'reason': 'Paper identity and full extracted-text coverage checked',
                'evidence': {'pdf': result.get('pdf_input'), 'reading_coverage': coverage}}
    if stage == 'FIND_RESOURCES':
        from src.repository_evidence import normalize_repository_url
        resources = result.get('resources') or {}
        selected = normalize_repository_url(resources.get('code_repo_url') or '')
        requested = normalize_repository_url(data.get('preferred_repo_url') or data.get('code_repo_url') or '')
        return {'success': bool(selected) and (not requested or requested.casefold() == selected.casefold()),
                'reason': 'Candidate URL checked; authorship and execution still require source evidence',
                'evidence': {'selected_repository': selected, 'requested_repository': requested}}
    return {'success': None, 'reason': 'Stage requires its own execution validator'}


@contextmanager
def _meter_agent(agent, session):
    previous = getattr(agent, 'llm', None)
    if previous is not None:
        agent.llm = session.wrap_llm(previous)
    try:
        yield
    finally:
        if previous is not None:
            agent.llm = previous


def _pdf_evidence(data, failed_result):
    from src.pdf_input import extract_pdf_input
    document = extract_pdf_input(data['pdf_path'])  # existing complete parser/fallback chain
    expected = ((data.get('pdf_input') or {}).get('sha256')
                or ((failed_result or {}).get('pdf_input') or {}).get('sha256'))
    if expected and expected != document.sha256:
        raise InvestigationStopped('Original PDF changed during recovery')
    coverage = (failed_result or {}).get('reading_coverage') or data.get('reading_coverage') or {}
    spans = coverage.get('spans') or []
    page_coverage, offset = [], 0
    for number, page in enumerate(document.page_texts, 1):
        end = offset + len(page)
        covered = any(isinstance(span, (list, tuple)) and len(span) == 2
                      and type(span[0]) is int and type(span[1]) is int
                      and span[0] <= offset and span[1] >= end for span in spans)
        page_coverage.append({'page': number, 'readable': bool(page.strip()),
                              'covered': covered, 'sha256': hashlib.sha256(page.encode('utf-8')).hexdigest()})
        offset = end + 1
    return document, {'pdf_sha256': document.sha256, 'pages': page_coverage,
                      'extraction': document.extraction, 'reading_coverage': coverage,
                      'repository_links': list(document.repository_links)}


def run_stage(agent, stage, data):
    """One evidenced correction per early stage, charged to the task ledger.

    Disabled/API legacy runs follow the original path. A failed correction is
    retained as failed; completion of a model call never verifies recovery.
    """
    session = current_investigation()
    if session is None:
        return agent.run(data)
    with session.phase(PHASES.get(stage, stage.lower())):
        session.ensure_running()
        _check_original_pdf(data)
        first = None
        try:
            with _meter_agent(agent, session):
                first = agent.run(data)
            _check_original_pdf(data, first)
            if stage not in {'READ_PAPER', 'FIND_RESOURCES'} or not data.get('pdf_path'):
                return first
            verification = _verify(stage, first, data)
            if verification['success']:
                session.remember_fact(stage, verification['evidence'])
                return first
            raise StageEvidenceError('Paper reading is incomplete' if stage == 'READ_PAPER'
                                     else 'No consistent repository candidate was obtained', first)
        except Exception as exc:
            session.ensure_running()
            from src.repository_repairer import cleanup_is_unconfirmed, repair_is_stopped
            exception_execution = getattr(exc, 'execution', {})
            if (isinstance(exc, InvestigationStopped) or getattr(exc, 'timed_out', False)
                    or getattr(exc, 'process_cleanup_confirmed', None) is False
                    or cleanup_is_unconfirmed(data) or repair_is_stopped(data)
                    or repair_is_stopped(exception_execution)):
                session.stop('execution_stopped_or_cleanup_unconfirmed',
                    cleanup_confirmed=False if (cleanup_is_unconfirmed(data)
                        or cleanup_is_unconfirmed(exception_execution)
                        or getattr(exc, 'process_cleanup_confirmed', None) is False) else None)
                raise
            problem = session.safe(str(exc))
            if stage not in {'READ_PAPER', 'FIND_RESOURCES'} or not data.get('pdf_path'):
                session.record_issue(problem, evidence={'stage': stage}, role=agent.name)
                raise
            # Authentication/transport failures have no new document evidence
            # that can repair the configured provider.
            if 'API' in problem and ('调用失败' in problem or 'Error' in problem):
                session.record_issue(problem, evidence={'stage': stage}, role=agent.name)
                raise
            if first is None and isinstance(getattr(exc, 'reading_coverage', None), dict):
                first = {'reading_coverage': deepcopy(exc.reading_coverage)}
            document, evidence = _pdf_evidence(data, first)
            issue = session.record_issue(problem, evidence=evidence, role=agent.name)
            session.remember_fact('original_pdf', {'sha256': document.sha256,
                'page_count': document.page_count, 'extraction': document.extraction})
            feedback = {'agent': agent.name, 'problem': problem, 'evidence': evidence,
                'required_checks': ['Read every missing page and its table headers/footnotes',
                    'Recheck original URI annotations and explicit user repository',
                    'Preserve the requested method, dataset, splits and experiment settings'],
                'issue_id': issue.get('id')}
            retry_data = {**data, 'retry_feedback': feedback}
            if stage == 'FIND_RESOURCES':
                retry_data.update(raw_text=document.text,
                    extracted_repository_links=list(document.repository_links),
                    extracted_code_urls=list(dict.fromkeys(link['url'] for link in document.repository_links)))
            from src.agent_collaboration import current_collaboration
            collaboration = current_collaboration()
            if collaboration and session.mode == 'collaborative':
                collaboration.send('Coordinator', agent.name, 'recovery_request', feedback)

            def recover():
                with _meter_agent(agent, session):
                    result = agent.run(retry_data)
                _check_original_pdf(data, result)
                # Preserve first-pass model usage in the original caller's accounting.
                if isinstance(result, dict):
                    result = deepcopy(result)
                    result['llm_calls'] = int(result.get('llm_calls') or 0) + int((first or {}).get('llm_calls') or 0)
                return result

            result = session.run_action('reread_paper' if stage == 'READ_PAPER' else 'rediscover_repository',
                recover, problem=problem, evidence=evidence, role=agent.name,
                review={'status': 'accepted', 'reviewer': 'original_source_rules',
                        'reason': 'Only the existing reader/finder may reread the unchanged original PDF'},
                verify=lambda value: _verify(stage, value, data))
            verified = _verify(stage, result, data)
            if not verified['success']:
                session.stop('no_progress', requires_human=True)
                raise StageEvidenceError(problem + '; evidence correction did not resolve the stage', result)
            session.remember_fact(stage, verified['evidence'])
            return result

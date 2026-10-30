"""Portable evidence records and content hashes for an exported report.

The manifest describes the supplied archive bytes. It does not certify paper
acceptance or imply that referenced datasets, sources or weights are included.
"""
import hashlib
import json

from src.author_claim_comparison import _redact_evidence
from src.report_revisions import sanitize_report_result
from src.result_conclusion import build_reproduction_conclusion


_EVIDENCE_FIELDS = (
    'paper_title', 'paper_info', 'pdf_input', 'pdf_resolution', 'reading_coverage',
    'resources', 'repository', 'spec_sha256', 'experiment_spec', 'experiment_request',
    'experiment_protocol', 'grounded_repository_plan', 'source_packet', 'execution_plan',
    'env_config', 'execution', 'validation', 'strict_validation',
    'strict_execution_verification', 'dataset_provenance', 'runtime_dataset_evidence',
    'source_repair', 'intelligent_recovery', 'recovery_summary', 'outcome_diagnosis', 'optimization', 'audit_stats',
    'environment_cleanup', 'run_finalization', 'run_dir', 'repository_executed',
    'run_status', 'execution_source', 'reproduction_scope', 'interrupted',
    'cancelled', 'timed_out',
)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def report_evidence_bytes(result, report, report_name):
    """Copy selected saved evidence; ignore a stale derived conclusion."""
    if not isinstance(result, dict) or not isinstance(result.get('data'), dict):
        raise ValueError('报告证据缺少结构化结果')
    safe = _redact_evidence(sanitize_report_result(result), '')
    data = safe['data']
    evidence = {key: data[key] for key in _EVIDENCE_FIELDS if key in data}
    evidence['reproduction_conclusion'] = build_reproduction_conclusion(data)
    from src.report_view_model import build_report_view_model
    evidence['report_view_model'] = build_report_view_model(data)
    return json_bytes({
        'schema_version': 1,
        'kind': 'autoreproducer_report_evidence',
        'state': safe.get('state'),
        'error': safe.get('error'),
        'report_export': safe.get('report_export'),
        'report': {'filename': report_name,
                   'sha256': hashlib.sha256(report.encode('utf-8')).hexdigest()},
        'scope': 'Saved result evidence. Referenced source, datasets and model weights are not bundled.',
        'data': evidence,
    })


def report_manifest_bytes(members, report_name, evidence_name=None):
    """Hash every payload member; the manifest itself is excluded."""
    return json_bytes({
        'schema_version': 1,
        'kind': 'autoreproducer_report_bundle',
        'report': report_name,
        'evidence': evidence_name,
        'hash_algorithm': 'sha256',
        'scope': 'Hashes verify bundled bytes, not scientific validity. The manifest excludes itself.',
        'files': [{'path': name, 'bytes': len(raw),
                   'sha256': hashlib.sha256(raw).hexdigest()}
                  for name, raw in sorted(members.items())],
    })

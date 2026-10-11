"""Check explicit paper split percentages against immutable, real mask arrays."""
import csv
import io
import json
import re

from src.safety.paths import workspace_path


_RATIO = re.compile(
    r'(?P<train>\d+(?:\.\d+)?)\s*%\s*,\s*'
    r'(?P<validation>\d+(?:\.\d+)?)\s*%\s*,?\s*(?:and\s+)?'
    r'(?P<test>\d+(?:\.\d+)?)\s*%\s+for\s+'
    r'train(?:ing)?\s*,\s*validation\s*,?\s*(?:and\s+)?test(?:ing)?', re.I)
_MASKS = {'train': 'train_mask', 'validation': 'val_mask', 'test': 'test_mask'}


def verify_partition(plan, packet, workspace):
    """Only claim arithmetic coverage for recognized paper/NPZ formats.

    This does not infer arbitrary loaders, nor equate filenames with ratios.
    The semantic audit must still establish that these files are actually used.
    """
    from src.discovered_repository_plan import PlanEvidenceError, _digest, _read_dataset_summary
    import numpy as np

    protocol = packet.get('experiment_protocol')
    if not protocol:
        return {'status': 'not_checked', 'reason': 'No frozen experiment protocol.'}
    citation_ids = protocol['dataset']['citations']
    declarations = []
    for citation in protocol['citations']:
        if citation['id'] not in citation_ids or citation['source'] != 'pdf':
            continue
        text = re.sub(r'\s+', ' ', citation['quote'])
        for match in _RATIO.finditer(text):
            prefix = text[max(0, match.start() - 140):match.start()]
            declarations.append({'percentages': {key: float(match[key]) for key in _MASKS},
                'basis': 'per_class' if re.search(r'(?:each|every) class', prefix, re.I) else 'global',
                'citation': citation['id'], 'quote': text[max(0, match.start()-140):match.end()]})
    if not declarations:
        return {'status': 'not_checked', 'reason': 'No recognized explicit paper percentage declaration.'}
    distinct = {(tuple(item['percentages'].values()), item['basis']) for item in declarations}
    if len(distinct) != 1:
        raise PlanEvidenceError('Selected paper evidence has ambiguous split percentage declarations',
            category='missing_information', planning_evidence={'stage': 'partition_validation', 'declarations': declarations})
    declaration = declarations[0]
    ratios = declaration['percentages']
    if any(not 0 < value < 100 for value in ratios.values()) or abs(sum(ratios.values()) - 100) > 1e-8:
        raise PlanEvidenceError('Paper percentage partition does not cover exactly the complete dataset',
            category='protocol_conflict', planning_evidence={'stage': 'partition_validation', 'declaration': declaration})
    paths = [name for source in plan['datasets'] if source['kind'] == 'bundled' for name in source['paths']]
    rows = []
    for name in paths:
        entry = packet['file_inventory'][name]
        evidence = entry.get('dataset_evidence')
        if not evidence or not name.lower().endswith('.npz'):
            continue
        summary = _read_dataset_summary(workspace, name, entry['sha256'])
        if summary != evidence:
            raise PlanEvidenceError('Split array evidence changed after discovery')
        arrays = json.loads(summary['text'])['arrays']
        if not set(_MASKS.values()) <= set(arrays):
            continue
        path = workspace_path(workspace, name, 'paper percentage split', must_exist=True, forbid_git=True)
        with np.load(path, allow_pickle=False) as data:
            masks = [data[key] for key in _MASKS.values()]
            if any(mask.ndim != 1 or mask.dtype.kind not in 'biu' or not np.isin(mask, [0, 1]).all() for mask in masks):
                raise PlanEvidenceError('Percentage split needs actual one-dimensional boolean or 0/1 masks')
            lengths = {len(mask) for mask in masks}
            if len(lengths) != 1 or not masks[0].size:
                raise PlanEvidenceError('Percentage split masks have inconsistent full-data sizes')
            coverage = sum(mask.astype(np.int8) for mask in masks)
            row = {'path': name, 'sha256': entry['sha256'], 'total': len(masks[0]),
                   'counts': {key: int(mask.sum()) for key, mask in zip(_MASKS, masks)}}
            if not np.all(coverage == 1):
                raise PlanEvidenceError('Paper percentage partition has overlapping or unassigned samples',
                    category='protocol_conflict', planning_evidence={'stage': 'partition_validation',
                        'declaration': declaration, 'observed': row})
        rows.append(row)
    if not rows:
        return {'status': 'not_checked', 'declaration': declaration,
                'reason': 'Selected data has no supported complete NPZ mask partition; semantic review still required.'}
    totals = {row['total'] for row in rows}
    if len(totals) != 1:
        raise PlanEvidenceError('Selected experiment split files have different full-data sizes',
            category='protocol_conflict', planning_evidence={'stage': 'partition_validation', 'observed': rows})
    total = rows[0]['total']
    class_sources = []
    label_vector = None
    if declaration['basis'] == 'per_class':
        # Obtain rounding bounds from actual complete labels, never an LLM's
        # guessed class count or a permissive percentage tolerance.
        for name in paths:
            entry = packet['file_inventory'][name]
            if not name.lower().endswith('.txt') or entry['bytes'] > 2 * 1024 * 1024:
                continue
            path = workspace_path(workspace, name, 'partition labels', must_exist=True, forbid_git=True)
            raw = path.read_bytes()
            if _digest(raw) != entry['sha256']:
                raise PlanEvidenceError('Full labels changed after discovery')
            try:
                reader = csv.DictReader(io.StringIO(raw.decode('utf-8')), delimiter='\t')
                if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                        or not {'node_id', 'label'} <= set(reader.fieldnames)):
                    continue
                records = list(reader)
                ids = [int(record['node_id']) for record in records]
                labels = [int(record['label']) for record in records]
                if len(ids) != total or sorted(ids) != list(range(total)) or min(labels) < 0:
                    continue
            except (ValueError, TypeError, UnicodeError, KeyError):
                continue
            vector = np.asarray(labels, dtype=np.int64)[np.argsort(ids)]
            if label_vector is not None and not np.array_equal(label_vector, vector):
                raise PlanEvidenceError('Selected complete label files disagree on class assignments',
                    category='missing_information', planning_evidence={'stage': 'partition_validation',
                        'declaration': declaration, 'class_sources': class_sources})
            label_vector = vector
            class_sources.append({'path': name, 'sha256': entry['sha256'], 'classes': len(set(labels))})
        counts = {source['classes'] for source in class_sources}
        if len(counts) != 1:
            raise PlanEvidenceError('Per-class split arithmetic needs unambiguous complete labels to bound rounding',
                category='missing_information', planning_evidence={'stage': 'partition_validation',
                    'declaration': declaration, 'observed': rows, 'class_sources': class_sources})
        # Two rounded allocations per class can affect the residual partition
        # by at most two nodes per class. This bound is fixed before execution.
        rounding_bound = 2 * next(iter(counts))
    else:
        rounding_bound = 2
    for row in rows:
        row['expected_counts'] = {key: total * value / 100 for key, value in ratios.items()}
        if label_vector is not None:
            path = workspace_path(workspace, row['path'], 'per-class split', must_exist=True, forbid_git=True)
            if _digest(path.read_bytes()) != row['sha256']:
                raise PlanEvidenceError('Split changed during per-class verification')
            with np.load(path, allow_pickle=False) as data:
                row['per_class_counts'] = [{
                    'label': int(label), 'total': int((label_vector == label).sum()),
                    'counts': {key: int(data[mask][label_vector == label].sum()) for key, mask in _MASKS.items()}
                } for label in np.unique(label_vector)]
            if any(abs(item['counts'][key] - item['total'] * ratios[key] / 100) > 2 + 1e-8
                   for item in row['per_class_counts'] for key in ratios):
                raise PlanEvidenceError('Actual per-class split counts conflict with paper percentages beyond integer rounding: '
                    + row['path'], category='protocol_conflict', planning_evidence={
                        'stage': 'partition_validation', 'declaration': declaration, 'observed': row,
                        'rounding_bound_samples_per_class': 2, 'class_sources': class_sources})
        if any(abs(row['counts'][key] - expected) > rounding_bound + 1e-8
               for key, expected in row['expected_counts'].items()):
            raise PlanEvidenceError('Actual split counts conflict with paper percentages beyond integer rounding: '
                + row['path'], category='protocol_conflict', planning_evidence={
                    'stage': 'partition_validation', 'declaration': declaration, 'observed': row,
                    'rounding_bound_samples': rounding_bound, 'class_sources': class_sources})
    return {'status': 'verified', 'declaration': declaration, 'observed': rows,
            'rounding_bound_samples': rounding_bound, 'class_sources': class_sources}

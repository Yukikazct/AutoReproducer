"""Lossless label layouts, including actual invented author subprocess runs."""
import hashlib
import io
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.discovered_repository_runtime import verify_capture, verify_label_layout_receipts
from test_discovered_repository_runtime import run_fixture
from test_discovered_repository_series import run_series


@pytest.fixture
def support(monkeypatch):
    pytest.importorskip('torch')
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'src'))
    spec = importlib.util.spec_from_file_location('_label_capture_test', root / 'src/experiments/discovered_capture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind', ['torch', 'numpy'])
@pytest.mark.parametrize('column', [False, True])
def test_labels_normalize_without_changing_author_storage(support, kind, column):
    import torch
    values = np.array([0, 1, 1, 0], dtype=np.int32)
    original = values.reshape(-1, 1) if column else values
    if kind == 'torch':
        original = torch.from_numpy(original.copy())
    labels, raw, evidence = support.normalize_author_labels(original)
    assert labels.dtype == torch.int64 and labels.tolist() == [0, 1, 1, 0]
    assert evidence['original_shape'] == ([4, 1] if column else [4])
    assert evidence['transformation'] == ('squeeze_column' if column else 'identity')
    tensors = {'labels': labels, 'raw_labels': raw, 'label_normalization': evidence}
    assert support.verify_label_normalization(tensors, {'label_normalization': evidence}) == evidence
    raw.reshape(-1)[0] = 1
    assert (original.reshape(-1)[0].item() if kind == 'torch' else original.reshape(-1)[0]) == 0
    with pytest.raises(ValueError, match='differs'):
        support.verify_label_normalization(tensors, {'label_normalization': evidence})


@pytest.mark.parametrize('values', [
    np.array([0., 1.]), np.array([True, False]), np.array([[0, 1], [1, 0]]),
    np.array([[[0]], [[1]]]), np.array([], dtype=np.int64),
    np.array([2**63], dtype=np.uint64),
])
def test_invalid_labels_cannot_be_coerced_to_valid_classes(support, values):
    with pytest.raises(ValueError, match='integer'):
        support.normalize_author_labels(values)


def test_uint_labels_are_lossless_and_index_shape_rules_remain_separate(support):
    import torch
    labels, _, _ = support.normalize_author_labels(np.array([[0], [1]], dtype=np.uint32))
    logits = torch.tensor([[2., 0.], [0., 2.]])
    assert support.select(logits, labels, torch.tensor([0, 1]), 2)[1].tolist() == [0, 1]
    for indices in (torch.tensor([[0], [1]]), torch.tensor([0, 0]), torch.tensor([0, 2])):
        with pytest.raises(ValueError):
            support.select(logits, labels, indices, 2)
    with pytest.raises(ValueError):
        support.select(logits, labels, None, 1)
    with pytest.raises(ValueError):
        support.select(logits, torch.tensor([0, 2]), None, 2)
    with pytest.raises(ValueError):
        support.select(logits[:1], labels, None, 2)


@pytest.mark.parametrize('dtype', ['|i1', '|u1'] + [
    endian + code + str(size) for endian in ('<', '>') for code in ('i', 'u') for size in (2, 4, 8)])
@pytest.mark.parametrize('column', [False, True])
def test_all_integer_dtypes_and_byte_orders_survive_actual_artifact_roundtrip(support, dtype, column):
    import torch
    values = np.array([0, 1, 3, 2], dtype=dtype)
    if column:
        values = values.reshape(-1, 1)
    labels, raw, metadata = support.normalize_author_labels(values)
    assert raw.dtype == torch.uint8 and raw.ndim == 1
    assert raw.numpy().tobytes() == values.tobytes(order='C')
    assert metadata['original_numpy_dtype'] == np.dtype(dtype).str
    assert metadata['original_sha256'] == hashlib.sha256(values.tobytes(order='C')).hexdigest()
    stream = io.BytesIO()
    torch.save({'labels': labels, 'raw_labels': raw, 'label_normalization': metadata}, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location='cpu', weights_only=True)
    assert support.verify_label_normalization(restored, {'label_normalization': metadata}) == metadata
    assert restored['labels'].tolist() == [0, 1, 3, 2]
    # Exact original bytes remain independent from the author array.
    values.reshape(-1)[0] = 7
    assert restored['raw_labels'].numpy().tobytes() == raw.numpy().tobytes()


@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'])
def test_torch_integer_dtypes_are_portably_serialized_as_original_bytes(support, dtype):
    import torch
    original = torch.from_numpy(np.array([[0], [1]], dtype=dtype))
    labels, raw, metadata = support.normalize_author_labels(original)
    stream = io.BytesIO()
    torch.save({'labels': labels, 'raw_labels': raw, 'label_normalization': metadata}, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    assert support.verify_label_normalization(restored, {'label_normalization': metadata}) == metadata
    assert original.dtype == getattr(torch, dtype) and original.shape == (2, 1)


def test_raw_label_storage_rejects_dtype_shape_bytes_and_overflow_tampering(support):
    import torch
    labels, raw, metadata = support.normalize_author_labels(np.array([[0], [1]], dtype=np.uint64))
    for changed in (raw[:-1], raw.to(torch.int8), raw.reshape(2, 8), torch.cat([raw, raw])):
        with pytest.raises(ValueError, match='byte storage'):
            support.verify_label_normalization(
                {'labels': labels, 'raw_labels': changed, 'label_normalization': metadata},
                {'label_normalization': metadata})
    for key, value in (('original_numpy_dtype', '<f8'), ('original_numpy_dtype', 'object'),
                       ('original_shape', [2, 2]), ('original_shape', [True, 1])):
        changed = {**metadata, key: value}
        with pytest.raises(ValueError, match='byte storage'):
            support.verify_label_normalization(
                {'labels': labels, 'raw_labels': raw, 'label_normalization': changed},
                {'label_normalization': changed})
    overflowing = torch.from_numpy(np.frombuffer(np.array([[0], [2**63]], dtype=np.uint64).tobytes(),
                                                dtype=np.uint8).copy())
    with pytest.raises(ValueError, match='overflow'):
        support.verify_label_normalization(
            {'labels': labels, 'raw_labels': overflowing, 'label_normalization': metadata},
            {'label_normalization': metadata})
    maximum = np.array([2**63 - 1], dtype=np.uint64)
    assert support.normalize_author_labels(maximum)[0].item() == 2**63 - 1


def test_normalized_values_and_layout_receipts_must_both_match(support):
    import torch
    labels, raw, metadata = support.normalize_author_labels(torch.tensor([[0], [1]]))
    for field, value in [('original_shape', [2]), ('original_sha256', '0' * 64),
                         ('transformation', 'identity'), ('normalized_dtype', 'torch.float32')]:
        bad = {**metadata, field: value}
        with pytest.raises(ValueError, match='differs'):
            support.verify_label_normalization(
                {'labels': labels, 'raw_labels': raw, 'label_normalization': bad}, {'label_normalization': bad})
    with pytest.raises(ValueError, match='differs'):
        support.verify_label_normalization(
            {'labels': 1 - labels, 'raw_labels': raw, 'label_normalization': metadata},
            {'label_normalization': metadata})
    with pytest.raises(ValueError, match='incomplete'):
        support.verify_label_normalization({'labels': labels, 'raw_labels': raw}, {})
    assert support.verify_label_normalization({'labels': labels}, {}) is None
    with pytest.raises(ValueError, match='Legacy'):
        support.verify_label_normalization({'labels': raw}, {})


def test_host_reconciles_shape_dtype_hash_and_independent_label_receipts():
    metadata = {'version': 2, 'original_shape': [12, 1], 'original_dtype': 'torch.int64',
                'original_numpy_dtype': '<i8',
                'original_sha256': 'a' * 64, 'normalized_shape': [12],
                'normalized_dtype': 'torch.int64', 'normalized_sha256': 'a' * 64,
                'transformation': 'squeeze_column'}
    verify_label_layout_receipts({'total_samples': 12, 'label_normalization': metadata},
                                {'label_normalization': metadata})
    for field, value in [('version', True), ('original_shape', [12, 2]),
                         ('original_dtype', 'torch.float32'), ('normalized_shape', [11]),
                         ('normalized_sha256', 'b' * 64), ('original_sha256', 'invalid'),
                         ('transformation', 'identity')]:
        bad = {**metadata, field: value}
        with pytest.raises(ValueError, match='lossless'):
            verify_label_layout_receipts({'total_samples': 12, 'label_normalization': bad},
                                        {'label_normalization': bad})
    with pytest.raises(ValueError, match='disagree'):
        verify_label_layout_receipts({'total_samples': 12, 'label_normalization': metadata}, {})
    with pytest.raises(ValueError, match='lossless'):
        verify_label_layout_receipts({'total_samples': 11, 'label_normalization': metadata},
                                    {'label_normalization': metadata})


def expose_column_labels(fixture):
    path = fixture['workspace'] / 'train.py'
    source = path.read_text('utf-8')
    source = source.replace("print('Test Accuracy:', accuracy)",
                            "column_labels = labels.reshape(-1, 1)\nprint('Test Accuracy:', accuracy)")
    # Native series labels live inside the original complete training function.
    source = source.replace('    return accuracy, model',
                            '    column_labels = labels.reshape(-1, 1)\n    return accuracy, model')
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(c for c in fixture['proposal']['citations'] if c.get('path') == path.name)['quote'] = source
    fixture['proposal']['capture']['labels'] = 'column_labels'


@pytest.mark.parametrize('series', [False, True])
def test_column_labels_survive_training_export_and_fresh_numpy_evaluation(tmp_path, series):
    if series:
        fixture, plan, spec, execution = run_series(tmp_path, mutate=expose_column_labels)
    else:
        fixture, plan, spec, _, execution = run_fixture(tmp_path, mutate=expose_column_labels)
    result = verify_capture(plan, fixture['workspace'], execution, spec)
    assert result['independent_metrics_pass'] is True
    capture = result['capture']['runs'] if series else [result['capture']]
    evaluated = result['independent_metrics']['runs'] if series else [result['independent_metrics']]
    for trained, independent in zip(capture, evaluated):
        metadata = trained['label_normalization']
        assert metadata['original_shape'] == [12, 1] and metadata['normalized_shape'] == [12]
        assert metadata['transformation'] == 'squeeze_column'
        assert independent['label_normalization'] == metadata
    if series:
        assert [r['accuracy'] for r in evaluated] == [1., .75, 1.]
    else:
        assert result['independent_metrics']['accuracy'] == 1.

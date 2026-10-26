"""Result investigations can inspect shape gaps without unpickling row values."""
import hashlib
import io
import json
import struct
import stat
import sys
import zipfile

import numpy as np
import pytest

from src.intelligent_recovery import RecoverySession
from src.investigation_artifacts import numpy_artifact_metadata
from src.investigation_tools import InvestigationTools


def observed_tools(tmp_path, raw, name='result.npz'):
    path = tmp_path / name
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    packet = {'file_inventory': {name: {'sha256': sha, 'text_available': False}},
              'investigation_artifacts': {name: {'sha256': sha, 'public': True}}}
    return InvestigationTools(tmp_path, packet, {}, tmp_path)


def arrays(**values):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def test_array_inspection_exposes_prediction_count_gap_without_row_values(tmp_path, monkeypatch):
    raw = arrays(predictions=np.zeros((3, 2)), sample_ids=np.array([71281, 71282]),
                 targets=np.array(['private-target-A', 'private-target-B']))
    tools = observed_tools(tmp_path, raw)
    monkeypatch.setattr(np, 'load', lambda *_a, **_k: pytest.fail('Array data must never be loaded'))
    result = tools.execute('artifact_contract', {'paths': ['result.npz']})
    assert result['status'] == 'completed', result
    value = result['evidence'][0]['value']
    assert value['leading_dimensions'] == {'predictions': 3, 'sample_ids': 2, 'targets': 2}
    assert value['arrays']['predictions']['shape'] == [3, 2]
    assert value['sample_correspondence'].startswith('unknown')
    assert 'private-target-A' not in json.dumps(result) and '71281' not in json.dumps(result)


def test_object_array_is_described_without_loading_its_pickle(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    np.save(buffer, np.array([{'secret-target': 'unexposed'}], dtype=object), allow_pickle=True)
    tools = observed_tools(tmp_path, buffer.getvalue(), 'result.npy')
    monkeypatch.setattr(np, 'load', lambda *_a, **_k: pytest.fail('Pickled objects must not be loaded'))
    result = tools.execute('artifact_contract', {'paths': ['result.npy']})
    assert result['status'] == 'completed'
    assert result['evidence'][0]['value']['array']['data_read'] is False
    assert 'secret-target' not in json.dumps(result)


@pytest.mark.parametrize('member', ['../private.npy', '/absolute.npy', 'private/labels.npy', 'x\\y.npy', 'C:result.npy'])
def test_array_archive_rejects_nonflat_member_paths(member):
    buffer, array = io.BytesIO(), io.BytesIO()
    np.save(array, np.zeros(1))
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(member, array.getvalue())
    with pytest.raises(ValueError, match='member'):
        numpy_artifact_metadata(buffer.getvalue(), '.npz')


@pytest.mark.parametrize('header', [b"{'shape': (2**999,), 'descr': '<f8', 'fortran_order': False}\n",
    b"{'shape': (-1,), 'descr': '<f8', 'fortran_order': False}\n",
    b"{'shape': (True,), 'descr': '<f8', 'fortran_order': False}\n",
    b"{'shape': (2,), 'descr': '<f8', 'fortran_order': False, 'extra': 1}\n"])
def test_malformed_array_header_is_rejected_without_execution(tmp_path, header):
    raw = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header
    tools = observed_tools(tmp_path, raw, 'result.npy')
    assert tools.execute('artifact_contract', {'paths': ['result.npy']})['status'] == 'rejected'


def test_header_size_and_archive_member_count_are_bounded():
    with pytest.raises(ValueError, match='budget'):
        numpy_artifact_metadata(b'\x93NUMPY\x02\x00' + (65537).to_bytes(4, 'little'), '.npy')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for index in range(65):
            archive.writestr(str(index) + '.npy', b'')
    with pytest.raises(ValueError, match='budget'):
        numpy_artifact_metadata(buffer.getvalue(), '.npz')


@pytest.mark.parametrize('kind', ['duplicate', 'symlink'])
def test_archive_duplicate_or_symlink_members_are_rejected(kind):
    buffer, array = io.BytesIO(), io.BytesIO()
    np.save(array, np.zeros(1))
    with zipfile.ZipFile(buffer, 'w') as archive:
        if kind == 'duplicate':
            archive.writestr('same.npy', array.getvalue())
            with pytest.warns(UserWarning, match='Duplicate'):
                archive.writestr('same.npy', array.getvalue())
        else:
            member = zipfile.ZipInfo('link.npy')
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, array.getvalue())
    with pytest.raises(ValueError, match='Duplicate|member'):
        numpy_artifact_metadata(buffer.getvalue(), '.npz')


def test_result_gap_observation_changes_the_next_investigation_decision(tmp_path):
    tools = observed_tools(tmp_path, arrays(predictions=np.zeros(3), targets=np.zeros(2)))
    model_calls = []
    class Investigator:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            context = json.loads(prompt)
            model_calls.append(context)
            if not context['probe_results']:
                return json.dumps({'action': 'probe', 'reason': 'Check prediction and target counts first.',
                    'tool': 'artifact_contract', 'arguments': {'paths': ['result.npz']}, 'evidence_ids': ['validation']})
            result = context['probe_results'][0]['evidence'][0]
            assert result['value']['leading_dimensions'] == {'predictions': 3, 'targets': 2}
            return json.dumps({'action': 'needs_human', 'reason': 'Output dimensions differ; correspondence is not established.',
                               'evidence_ids': [result['id']]})
    session = RecoverySession(Investigator())
    session.attach({}, tmp_path)
    plan = {'repository': {'revision': 'a' * 40}, 'steps': [], 'datasets': []}
    execution = {'success': True, 'final': {'exit_code': 0}}
    validation = {'is_reproduced': None, 'paper_metrics_verified': False}
    result = session.investigate(trigger='result_discrepancy', plan=plan, packet=tools.packet,
        workspace=tmp_path, environment={}, execution=execution, validation=validation, tools=tools)
    assert result['status'] == 'needs_human' and session.state['usage']['probes'] == 1
    assert len(model_calls) == 2
    assert validation == {'is_reproduced': None, 'paper_metrics_verified': False}
    assert (tmp_path / 'result.npz').read_bytes() == arrays(predictions=np.zeros(3), targets=np.zeros(2))


def test_real_runner_declared_result_is_available_to_the_postrun_investigator(tmp_path):
    from src.repository_runner import RepositoryRunner
    workspace = tmp_path / 'author'
    workspace.mkdir()
    source = workspace / 'train.py'
    source.write_text("import shutil\nshutil.copyfile('input.npz', 'predictions.npz')\nprint('author command completed')\n")
    (workspace / 'input.npz').write_bytes(arrays(predictions=np.zeros(3), targets=np.zeros(2)))
    packet = {'file_inventory': {path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'text_available': path.suffix == '.py'} for path in workspace.iterdir()}}
    steps = [{'id': 'train', 'kind': 'train', 'argv': ['python', 'train.py'], 'cwd': '.',
              'artifacts': ['predictions.npz']}]
    plan = {'repository': {'revision': 'a' * 40}, 'steps': steps, 'datasets': []}
    execution = RepositoryRunner().run(workspace, steps, {'python_executable': sys.executable,
        'requirements_txt': '', 'auto_prepare': False})
    assert execution['success'], execution
    assert execution['steps'][0]['artifacts'][0]['matched'] is True
    class Investigator:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            context = json.loads(prompt)
            if not context['probe_results']:
                assert 'predictions.npz' in context['artifact_catalog']
                return json.dumps({'action': 'probe', 'reason': 'Inspect the actual declared result.',
                    'tool': 'artifact_contract', 'arguments': {'paths': ['predictions.npz']},
                    'evidence_ids': ['validation']})
            result = context['probe_results'][0]
            assert result['status'] == 'completed'
            assert result['evidence'][0]['value']['leading_dimensions']['predictions'] == 3
            return json.dumps({'action': 'needs_human', 'reason': 'The output headers do not establish row correspondence.',
                               'evidence_ids': [result['evidence'][0]['id']]})
    session = RecoverySession(Investigator())
    session.attach({}, tmp_path)
    result = session.investigate(trigger='result_discrepancy', plan=plan, packet=packet, workspace=workspace,
        environment={}, execution=execution, validation={'paper_metrics_verified': False})
    assert result['status'] == 'needs_human'
    assert session.state['usage']['probes'] == 1
    assert execution['success'] is True

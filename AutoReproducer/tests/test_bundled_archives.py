"""Original bundled ZIP data must be complete, fixed and unable to overwrite source."""
from copy import deepcopy
import hashlib
import stat
import zipfile

import pytest

from src.bundled_archives import inspect_bundled_zip, extract_bundled_zip
from src.discovered_repository_reproduction import prepare_plan_datasets
from src import discovered_repository_plan as planner
from discovered_fixtures import make_experiment


def archive_fixture(tmp_path, entries):
    with zipfile.ZipFile(tmp_path / 'inputs.zip', 'w') as archive:
        for name, raw in entries:
            info = name if isinstance(name, zipfile.ZipInfo) else zipfile.ZipInfo(name)
            if isinstance(name, str):
                info.filename = name  # Preserve hostile ZIP path bytes even on Windows.
            archive.writestr(info, raw)
    return hashlib.sha256((tmp_path / 'inputs.zip').read_bytes()).hexdigest()


def test_entire_original_archive_is_prepared_and_recorded(tmp_path):
    entries = [('dataset/train.txt', b'1 2 3'), ('dataset/test.txt', b'4 5 6')]
    digest = archive_fixture(tmp_path, entries)
    manifest = inspect_bundled_zip(tmp_path, 'inputs.zip', digest, '.')
    plan = {'datasets': [{'kind': 'bundled', 'paths': ['inputs.zip'], 'extraction': manifest}]}
    snapshot = {'files': {'inputs.zip': digest}, 'url': 'https://github.com/invented/fixture', 'resolved_sha': 'a' * 40}
    records = prepare_plan_datasets(tmp_path, tmp_path, plan, snapshot, offline=True)
    assert records['verified'] is True
    assert set(records['sources'][0]['extracted_files']) == {name for name, _ in entries}
    for name, raw in entries:
        assert (tmp_path / name).read_bytes() == raw
    with pytest.raises(ValueError, match='overwrite'):
        extract_bundled_zip(tmp_path, manifest)


@pytest.mark.parametrize('name', ['../outside.txt', 'C:/outside.txt', '.git/config', '_autorepro_capture/config.json', 'folder\\file.txt'])
def test_archive_member_cannot_escape_or_write_execution_metadata(tmp_path, name):
    digest = archive_fixture(tmp_path, [(name, b'input')])
    with pytest.raises(ValueError):
        inspect_bundled_zip(tmp_path, 'inputs.zip', digest, '.')


@pytest.mark.parametrize('names', [['data.txt', 'DATA.txt'], ['data', 'data/file.txt']])
def test_archive_cannot_have_duplicate_or_conflicting_paths(tmp_path, names):
    digest = archive_fixture(tmp_path, [(name, b'input') for name in names])
    with pytest.raises(ValueError, match='duplicate|collides'):
        inspect_bundled_zip(tmp_path, 'inputs.zip', digest, '.')


def test_all_members_are_checked_before_any_file_is_written(tmp_path):
    (tmp_path / 'train.py').write_text('original author code', 'utf-8')
    digest = archive_fixture(tmp_path, [('dataset/data.txt', b'input'), ('train.py', b'replacement')])
    with pytest.raises(ValueError, match='overwrite'):
        inspect_bundled_zip(tmp_path, 'inputs.zip', digest, '.')
    assert not (tmp_path / 'dataset').exists()
    assert (tmp_path / 'train.py').read_text('utf-8') == 'original author code'


def test_archive_and_frozen_member_hashes_must_match(tmp_path):
    digest = archive_fixture(tmp_path, [('dataset/data.txt', b'input')])
    manifest = inspect_bundled_zip(tmp_path, 'inputs.zip', digest, '.')
    changed = deepcopy(manifest)
    changed['files']['dataset/data.txt']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='frozen plan'):
        extract_bundled_zip(tmp_path, changed)
    archive_fixture(tmp_path, [('dataset/data.txt', b'replaced')])
    with pytest.raises(ValueError, match='pinned'):
        extract_bundled_zip(tmp_path, manifest)
    assert not (tmp_path / 'dataset').exists()


def test_symlink_member_is_rejected(tmp_path):
    info = zipfile.ZipInfo('data.txt')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    digest = archive_fixture(tmp_path, [(info, b'../outside')])
    with pytest.raises(ValueError, match='linked'):
        inspect_bundled_zip(tmp_path, 'inputs.zip', digest, '.')


def test_extraction_limit_is_checked_before_writing(tmp_path, monkeypatch):
    digest = archive_fixture(tmp_path, [('dataset/data.txt', b'123456')])
    monkeypatch.setattr('src.bundled_archives.MAX_BYTES', 5)
    with pytest.raises(ValueError, match='limit'):
        inspect_bundled_zip(tmp_path, 'inputs.zip', digest, '.')
    assert not (tmp_path / 'dataset').exists()


def test_inventory_has_selectable_ids_and_still_proves_only_file_existence(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    view = planner._planning_view(packet)
    identifier, ref = next((key, ref) for key, ref in view['evidence_excerpts'].items()
                          if ref['source'] == 'inventory' and ref['path'] == 'dataset.json')
    citation = planner._citations({'citations': [{'id': 'data_exists', 'evidence_id': identifier}]}, packet, fixture['workspace'])
    assert citation['data_exists']['evidence_scope'] == 'file_existence_only'
    with pytest.raises(ValueError, match='unknown source excerpt'):
        planner._citations({'citations': [{'id': 'data_exists', 'evidence_id': 'dataset.json'}]}, packet, fixture['workspace'])


def test_dataset_plan_compiles_archive_hashes_from_actual_bytes(tmp_path):
    fixture = make_experiment(tmp_path)
    workspace = fixture['workspace']
    digest = archive_fixture(workspace, [('new_data/train.txt', b'1 2 3')])
    fixture['snapshot']['files']['inputs.zip'] = digest
    packet = planner.build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    citations = planner._citations(fixture['proposal'], packet, workspace)
    dataset = {'kind': 'bundled', 'paths': ['inputs.zip'], 'extract_to': '.', 'citations': ['readme']}
    result = planner._datasets([dataset], citations, packet, workspace)
    assert result[0]['extraction']['archive_sha256'] == digest
    assert result[0]['extraction']['files']['new_data/train.txt']['bytes'] == 5

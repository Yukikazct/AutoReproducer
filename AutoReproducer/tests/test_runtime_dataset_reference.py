"""Invented ZIP fixtures check source comparison, never real-paper success."""
from copy import deepcopy
from io import BytesIO
import hashlib
import stat
import struct
import warnings
from urllib.error import URLError
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from src import runtime_dataset_reference as reference
from src import runtime_dataset_evidence as evidence


def fixture_source(class_name='CoraGraphDataset'):
    canonical, native = reference.DGL_NATIVE_DATASETS[class_name]
    loader = {'kind': 'author_runtime', 'framework': 'dgl.data', 'dataset_class': class_name,
              'dataset_name': canonical, 'framework_dataset_name': native, 'framework_version': '1.1.2'}
    members = {f'ind.{native}.{part}': f'invented {native} {part} bytes'.encode()
               for part in reference.RAW_PARTS}
    raw_path = f'C:\\invented\\{native}'
    files = {raw_path + '\\' + name: {'role': 'raw', 'bytes': len(body),
             'sha256': hashlib.sha256(body).hexdigest()} for name, body in members.items()}
    files[raw_path + '\\processed.bin'] = {'role': 'processed', 'bytes': 123, 'sha256': '0' * 64}
    observation = {'verified': True, 'declaration': deepcopy(loader), 'dataset_name': canonical,
        'framework_dataset_name': native, 'framework_version': '1.1.2',
        'source_url': f'https://data.dgl.ai/dataset/{native}.zip', 'raw_path': raw_path, 'files': files}
    return {'kind': 'author_runtime', 'verified': True, 'loader': loader,
            'observations': [observation]}, members


def zip_bytes(members):
    stream = BytesIO()
    with ZipFile(stream, 'w', compression=ZIP_DEFLATED) as archive:
        for name, body in members.items() if isinstance(members, dict) else members:
            archive.writestr(name, body)
    return stream.getvalue()


class Response(BytesIO):
    def __init__(self, body, url, headers=None, status=200):
        super().__init__(body)
        self.url, self.headers, self.status = url, headers or {}, status

    def geturl(self):
        return self.url


def fake_download(monkeypatch, body, *, final_url=None, headers=None, status=200):
    calls = []
    def open_canonical(url, timeout):
        calls.append((url, timeout))
        return Response(body, final_url or url, headers, status)
    monkeypatch.setattr(reference, '_open_canonical', open_canonical)
    return calls


@pytest.mark.parametrize('class_name', list(reference.DGL_NATIVE_DATASETS))
def test_all_supported_native_sources_compare_eight_raw_files_without_promoting_protocol(monkeypatch, class_name):
    source, members = fixture_source(class_name)
    original = deepcopy(source)
    body = zip_bytes(members)
    calls = fake_download(monkeypatch, body, headers={'Content-Length': str(len(body)),
        'ETag': 'current-server-tag', 'Last-Modified': 'Wed, 03 Jun 2020 07:07:32 GMT'})
    result = reference.audit_dgl_reference(source)
    assert result['status'] == 'verified' and result['verified'] is True
    assert result['current_source_match'] is True and not result['errors']
    assert result['published_checksum_verified'] is False
    assert result['paper_protocol_verified'] is False and result['processed_inputs_verified'] is False
    receipt = result['canonical_reference']
    assert receipt['archive_sha256'] == hashlib.sha256(body).hexdigest()
    assert receipt['archive_bytes'] == len(body) and receipt['fetched_at'].endswith('+00:00')
    assert receipt['etag'] == 'current-server-tag' and len(receipt['members']) == 8
    assert len(result['comparisons']) == 8 and all(item['matches'] for item in result['comparisons'])
    assert calls == [(source['observations'][0]['source_url'], 5)] and source == original


def test_repeated_observations_use_one_fresh_reference_and_compare_every_occurrence(monkeypatch):
    source, members = fixture_source()
    source['observations'].append(deepcopy(source['observations'][0]))
    calls = fake_download(monkeypatch, zip_bytes(members))
    result = reference.audit_dgl_reference(source)
    assert result['verified'] is True and len(result['comparisons']) == 16 and len(calls) == 1
    assert {item['observation_index'] for item in result['comparisons']} == {0, 1}


@pytest.mark.parametrize('field,value', [('sha256', '0' * 64), ('bytes', 999)])
def test_any_raw_digest_or_size_mismatch_stays_explicit_and_preserves_source(monkeypatch, field, value):
    source, members = fixture_source()
    raw = next(item for item in source['observations'][0]['files'].values() if item['role'] == 'raw')
    raw[field] = value
    original = deepcopy(source)
    fake_download(monkeypatch, zip_bytes(members))
    result = reference.audit_dgl_reference(source)
    assert result['status'] == 'mismatch' and result['current_source_match'] is False
    assert result['verified'] is False and sum(not item['matches'] for item in result['comparisons']) == 1
    assert source == original and result['paper_protocol_verified'] is False


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'traversal', 'absolute', 'backslash', 'nested'])
def test_zip_requires_exact_unique_root_member_names(monkeypatch, change):
    source, members = fixture_source()
    entries = list(members.items())
    name, body = entries[0]
    if change == 'missing': entries.pop()
    elif change == 'extra': entries.append(('unexpected', b'invented'))
    elif change == 'duplicate': entries[-1] = entries[0]
    else:
        prefixes = {'traversal': '../', 'absolute': '/', 'backslash': '..\\', 'nested': 'nested/'}
        entries[0] = (prefixes[change] + name, body)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        archive = zip_bytes(entries)
    fake_download(monkeypatch, archive)
    result = reference.audit_dgl_reference(source)
    assert result['status'] == 'unavailable' and result['current_source_match'] is None
    assert result['errors'][0]['stage'] == 'archive' and not result['comparisons']
    assert result['canonical_reference']['archive_sha256'] == hashlib.sha256(archive).hexdigest()


def test_symlink_archive_member_cannot_be_compared_as_regular_raw_bytes(monkeypatch):
    source, members = fixture_source()
    name = next(iter(members))
    link = ZipInfo(name)
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    entries = [(link if key == name else key, value) for key, value in members.items()]
    fake_download(monkeypatch, zip_bytes(entries))
    assert reference.audit_dgl_reference(source)['status'] == 'unavailable'


def test_raw_local_and_central_member_names_with_nul_suffix_are_not_silently_truncated(monkeypatch):
    source, members = fixture_source()
    name = next(iter(members))
    marker = name + 'Xhidden'
    entries = [(marker if key == name else key, body) for key, body in members.items()]
    archive = zip_bytes(entries)
    assert archive.count(marker.encode()) == 2
    archive = archive.replace(marker.encode(), (name + '\x00hidden').encode())
    with ZipFile(BytesIO(archive)) as parsed:
        first = parsed.infolist()[0]
        assert first.filename == name and first.orig_filename != first.filename
    fake_download(monkeypatch, archive)
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'archive' and result['current_source_match'] is None


@pytest.mark.parametrize('archive', [b'not a ZIP', b'PK\x03\x04truncated'])
def test_corrupt_or_truncated_archive_is_unavailable(monkeypatch, archive):
    source, _ = fixture_source()
    fake_download(monkeypatch, archive)
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'archive' and result['current_source_match'] is None


def test_encrypted_zip_flag_is_not_accepted_as_plain_raw_data(monkeypatch):
    source, members = fixture_source()
    archive = bytearray(zip_bytes(members))
    central = archive.index(b'PK\x01\x02')
    struct.pack_into('<H', archive, 6, struct.unpack_from('<H', archive, 6)[0] | 1)
    struct.pack_into('<H', archive, central + 8, struct.unpack_from('<H', archive, central + 8)[0] | 1)
    fake_download(monkeypatch, bytes(archive))
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'archive' and 'regular raw file' in result['errors'][0]['reason']


def test_member_crc_error_preserves_unknown_instead_of_matching_a_partial_read(monkeypatch):
    source, members = fixture_source()
    stream = BytesIO()
    with ZipFile(stream, 'w') as output:
        for name, body in members.items(): output.writestr(name, body)
    archive = bytearray(stream.getvalue())
    name_length, extra_length = struct.unpack_from('<HH', archive, 26)
    archive[30 + name_length + extra_length] ^= 1
    fake_download(monkeypatch, bytes(archive))
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'archive' and result['current_source_match'] is None
    assert not result['comparisons']


@pytest.mark.parametrize('change', ['not_verified', 'class', 'native', 'version', 'dataset',
    'no_observations', 'declaration', 'origin', 'missing_raw', 'raw_directory', 'invalid_hash', 'bool_size'])
def test_incomplete_or_inconsistent_observation_is_unknown_without_fetch(monkeypatch, change):
    source, _ = fixture_source()
    observation = source['observations'][0]
    raw = next(item for item in observation['files'].values() if item['role'] == 'raw')
    if change == 'not_verified': source['verified'] = False
    elif change == 'class': source['loader']['dataset_class'] = 'AuthorCoraSubclass'
    elif change == 'native': source['loader']['framework_dataset_name'] = 'citeseer'
    elif change == 'version': source['loader']['framework_version'] = '9.9'
    elif change == 'dataset': source['loader']['dataset_name'] = 'Pubmed'
    elif change == 'no_observations': source['observations'] = []
    elif change == 'declaration': observation['declaration']['function'] = 'another_loader'
    elif change == 'origin': observation['source_url'] = 'https://example.org/cora_v2.zip'
    elif change == 'missing_raw': observation['files'].pop(next(iter(observation['files'])))
    elif change == 'raw_directory': observation['raw_path'] += '/elsewhere'
    elif change == 'invalid_hash': raw['sha256'] = 'unknown'
    elif change == 'bool_size': raw['bytes'] = True
    def forbidden(*args): raise AssertionError('Invalid observations must not start reference downloads')
    monkeypatch.setattr(reference, '_open_canonical', forbidden)
    result = reference.audit_dgl_reference(source)
    assert result['status'] == 'unavailable' and result['current_source_match'] is None
    assert result['errors'][0]['stage'] == 'observation' and result['canonical_reference'] is None


def test_offline_source_preserves_unknown_and_author_completion_record(monkeypatch):
    source, _ = fixture_source()
    original = deepcopy(source)
    def offline(*args): raise URLError('invented offline network')
    monkeypatch.setattr(reference, '_open_canonical', offline)
    result = reference.audit_dgl_reference(source)
    assert result['status'] == 'unavailable' and result['current_source_match'] is None
    assert result['errors'][0]['stage'] == 'download' and 'offline' in result['errors'][0]['reason']
    assert source == original and source['verified'] is True


@pytest.mark.parametrize('url', ['http://data.dgl.ai/dataset/cora_v2.zip',
    'https://example.org/cora_v2.zip', 'https://data.dgl.ai/dataset/pubmed.zip',
    'https://data.dgl.ai/dataset/cora_v2.zip?token=invented'])
def test_redirect_is_rejected_before_following_a_different_url(url):
    handler = reference._CanonicalRedirect('https://data.dgl.ai/dataset/cora_v2.zip')
    with pytest.raises(ValueError, match='redirect changed'):
        handler.redirect_request(None, None, 302, 'Found', {}, url)


def test_unexpected_final_url_is_rejected_before_body_read(monkeypatch):
    source, members = fixture_source()
    fake_download(monkeypatch, zip_bytes(members), final_url='https://example.org/cora_v2.zip')
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'download' and result['current_source_match'] is None


@pytest.mark.parametrize('headers', [{'Content-Length': '999999999999'}, {'Content-Length': '-1'},
    {'Content-Length': 'unknown'}, {'Content-Length': '1'}])
def test_invalid_or_truncated_http_length_is_unknown(monkeypatch, headers):
    source, members = fixture_source()
    fake_download(monkeypatch, zip_bytes(members), headers=headers)
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'download' and not result['comparisons']


@pytest.mark.parametrize('status', [204, 206])
def test_incomplete_http_success_is_not_a_complete_canonical_snapshot(monkeypatch, status):
    source, members = fixture_source()
    fake_download(monkeypatch, zip_bytes(members), status=status)
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'download' and result['current_source_match'] is None


def test_download_has_an_elapsed_time_bound(monkeypatch):
    source, members = fixture_source()
    fake_download(monkeypatch, zip_bytes(members))
    ticks = iter([0, 31])
    monkeypatch.setattr(reference.time, 'monotonic', lambda: next(ticks))
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'download' and 'time bound' in result['errors'][0]['reason']


@pytest.mark.parametrize('delayed_read', ['body', 'eof'])
def test_download_finishing_its_last_read_after_the_deadline_is_not_verified(monkeypatch, delayed_read):
    source, members = fixture_source()
    body = zip_bytes(members)
    clock = [0]
    class SlowResponse(Response):
        def read1(self, size):
            chunk = super().read1(size)
            if bool(chunk) == (delayed_read == 'body'):
                clock[0] = 61
            return chunk
    monkeypatch.setattr(reference.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(reference, '_open_canonical', lambda url, timeout: SlowResponse(body, url))
    result = reference.audit_dgl_reference(source)
    assert result['errors'][0]['stage'] == 'download' and 'time bound' in result['errors'][0]['reason']
    assert result['current_source_match'] is None and not result['comparisons']


@pytest.mark.parametrize('timeout', [0, -1, 61, float('nan'), float('inf'), True, '30'])
def test_invalid_timeouts_do_not_download(monkeypatch, timeout):
    source, _ = fixture_source()
    def forbidden(*args): raise AssertionError('Invalid timeouts must not start a download')
    monkeypatch.setattr(reference, '_open_canonical', forbidden)
    result = reference.audit_dgl_reference(source, timeout=timeout)
    assert result['current_source_match'] is None and 'timeout' in result['errors'][0]['reason']


def test_undeclared_stream_and_archive_member_bounds_are_enforced(monkeypatch):
    source, members = fixture_source()
    fake_download(monkeypatch, zip_bytes(members))
    monkeypatch.setattr(reference, 'MAX_ARCHIVE_BYTES', 20)
    assert reference.audit_dgl_reference(source)['errors'][0]['stage'] == 'download'
    monkeypatch.setattr(reference, 'MAX_ARCHIVE_BYTES', 32 * 1024 * 1024)
    monkeypatch.setattr(reference, 'MAX_MEMBER_BYTES', 5)
    assert reference.audit_dgl_reference(source)['errors'][0]['stage'] == 'archive'
    monkeypatch.setattr(reference, 'MAX_MEMBER_BYTES', 64 * 1024 * 1024)
    monkeypatch.setattr(reference, 'MAX_RAW_BYTES', 5)
    assert reference.audit_dgl_reference(source)['errors'][0]['stage'] == 'archive'


def test_no_local_file_open_or_pickle_load_is_needed(monkeypatch):
    source, members = fixture_source()
    fake_download(monkeypatch, zip_bytes(members))
    def forbidden(*args, **kwargs): raise AssertionError('Reference audit must not read or mutate author files')
    monkeypatch.setattr('builtins.open', forbidden)
    monkeypatch.setattr('pickle.loads', forbidden)
    assert reference.audit_dgl_reference(source)['verified'] is True


def mock_native_verification(monkeypatch, sources):
    """Isolate orchestration from the separately tested native file verifier."""
    calls = []
    def verify(plan, receipts, workspace):
        calls.append((deepcopy(plan), deepcopy(receipts), workspace))
        assert len(plan['datasets']) == 1
        declaration = plan['datasets'][0]
        return [deepcopy(source) for source in sources if source['loader'] == declaration]
    monkeypatch.setattr(evidence, 'verify_runtime_datasets', verify)
    return calls


def assert_no_strict_dataset_claims(audit):
    assert audit['published_checksum_verified'] is False
    assert audit['paper_protocol_verified'] is False
    assert audit['independent_reference_verified'] is False


def test_runtime_audit_defaults_to_observed_inputs_without_reference_download(monkeypatch, tmp_path):
    source, _ = fixture_source()
    plan = {'datasets': [deepcopy(source['loader'])]}
    receipts = [{'status': 'completed', 'native_datasets': deepcopy(source['observations'])}]
    original = deepcopy((plan, receipts, source))
    verification_calls = mock_native_verification(monkeypatch, [source])
    def forbidden(*args): raise AssertionError('Default audits must not download a reference')
    monkeypatch.setattr(reference, '_open_canonical', forbidden)

    result = evidence.audit_runtime_datasets(plan, receipts, tmp_path)

    assert result['status'] == 'verified' and result['verified'] is True
    assert result['basis'] == 'native_constructor_and_unchanged_input_files'
    assert result['canonical_source_match_verified'] is False
    assert result['sources'] == [source] and not result['errors']
    assert 'canonical_reference_audit' not in result['sources'][0]
    assert verification_calls == [(plan, receipts, tmp_path)]
    assert (plan, receipts, source) == original
    assert_no_strict_dataset_claims(result)


def test_runtime_audit_opt_in_compares_verified_dgl_bytes_without_promoting_strict_claims(monkeypatch, tmp_path):
    source, members = fixture_source()
    plan = {'datasets': [deepcopy(source['loader'])]}
    receipts = [{'status': 'completed', 'native_datasets': deepcopy(source['observations'])}]
    original = deepcopy((plan, receipts, source))
    mock_native_verification(monkeypatch, [source])
    download_calls = fake_download(monkeypatch, zip_bytes(members))

    result = evidence.audit_runtime_datasets(plan, receipts, tmp_path, compare_canonical_source=True)

    assert result['status'] == 'verified' and result['verified'] is True
    assert result['canonical_source_match_verified'] is True and not result['errors']
    reference_audit = result['sources'][0]['canonical_reference_audit']
    assert reference_audit['verified'] is True and reference_audit['current_source_match'] is True
    assert len(reference_audit['comparisons']) == 8
    assert all(item['matches'] for item in reference_audit['comparisons'])
    assert reference_audit['processed_inputs_verified'] is False
    assert download_calls == [(source['observations'][0]['source_url'], 5)]
    assert (plan, receipts, source) == original
    assert_no_strict_dataset_claims(result)
    assert reference_audit['published_checksum_verified'] is False
    assert reference_audit['paper_protocol_verified'] is False


def test_runtime_reference_mismatch_preserves_native_verification_and_never_promotes_provenance(monkeypatch, tmp_path):
    source, members = fixture_source()
    source['source'] = 'original_author_constructor_and_fixed_framework'
    raw = next(item for item in source['observations'][0]['files'].values() if item['role'] == 'raw')
    raw['sha256'] = '0' * 64
    plan = {'datasets': [deepcopy(source['loader'])]}
    receipts = [{'status': 'completed', 'native_datasets': deepcopy(source['observations'])}]
    original = deepcopy((plan, receipts, source))
    mock_native_verification(monkeypatch, [source])
    fake_download(monkeypatch, zip_bytes(members))

    result = evidence.audit_runtime_datasets(plan, receipts, tmp_path, compare_canonical_source=True)

    assert result['status'] == 'verified' and result['verified'] is True
    assert result['canonical_source_match_verified'] is False
    audited_source = result['sources'][0]
    assert audited_source['verified'] is True and audited_source['source'] == source['source']
    reference_audit = audited_source['canonical_reference_audit']
    assert reference_audit['status'] == 'mismatch' and reference_audit['verified'] is False
    assert reference_audit['current_source_match'] is False
    assert sum(not item['matches'] for item in reference_audit['comparisons']) == 1
    assert reference_audit['processed_inputs_verified'] is False
    assert reference_audit['published_checksum_verified'] is False
    assert reference_audit['paper_protocol_verified'] is False
    assert (plan, receipts, source) == original
    assert_no_strict_dataset_claims(result)


@pytest.mark.parametrize('include_dgl', [False, True])
def test_runtime_audit_requires_every_source_to_have_a_canonical_match(monkeypatch, tmp_path, include_dgl):
    dgl_source, members = fixture_source()
    other_source = {'kind': 'author_runtime', 'verified': True,
        'loader': {'kind': 'author_runtime', 'framework': 'torch_geometric.datasets',
                   'dataset_class': 'Planetoid', 'dataset_name': 'Cora'},
        'observations': []}
    sources = ([dgl_source] if include_dgl else []) + [other_source]
    plan = {'datasets': [deepcopy(source['loader']) for source in sources]}
    receipts = [{'status': 'completed'}]
    original = deepcopy((plan, receipts, sources))
    verification_calls = mock_native_verification(monkeypatch, sources)
    download_calls = fake_download(monkeypatch, zip_bytes(members))

    result = evidence.audit_runtime_datasets(plan, receipts, tmp_path, compare_canonical_source=True)

    assert result['status'] == 'verified' and result['verified'] is True
    assert len(result['sources']) == len(sources) and not result['errors']
    assert result['canonical_source_match_verified'] is False
    assert len(verification_calls) == len(sources)
    assert 'canonical_reference_audit' not in result['sources'][-1]
    if include_dgl:
        assert result['sources'][0]['canonical_reference_audit']['current_source_match'] is True
        assert download_calls == [(dgl_source['observations'][0]['source_url'], 5)]
    else:
        assert not download_calls
    assert (plan, receipts, sources) == original
    assert_no_strict_dataset_claims(result)


def test_failed_native_verification_cannot_start_a_reference_download(monkeypatch, tmp_path):
    source, _ = fixture_source()
    plan = {'datasets': [deepcopy(source['loader'])]}
    receipts = [{'native_datasets': [{'declaration': deepcopy(source['loader']),
                                    'verification_error': 'invented changed raw file'}]}]
    def failed(*args): raise ValueError('invented native verification failure')
    def forbidden(*args): raise AssertionError('Unverified native input must not download a reference')
    monkeypatch.setattr(evidence, 'verify_runtime_datasets', failed)
    monkeypatch.setattr(reference, '_open_canonical', forbidden)

    result = evidence.audit_runtime_datasets(plan, receipts, tmp_path, compare_canonical_source=True)

    assert result['status'] == 'unavailable' and result['verified'] is False
    assert result['sources'] == [] and result['canonical_source_match_verified'] is False
    assert result['errors'] == [{'loader': source['loader'],
        'reason': 'invented native verification failure',
        'observation_errors': ['invented changed raw file']}]
    assert_no_strict_dataset_claims(result)

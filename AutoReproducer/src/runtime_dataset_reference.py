"""Compare observed DGL raw bytes with a separately fetched current source ZIP.

This receipt establishes a current canonical-source match only. It does not
establish a published upstream checksum, paper protocol, processed graph
identity or independently reproduced model metrics. Author files are never
opened, extracted over, imported or unpickled here.
"""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import re
import stat
import time
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from src.experiments.observe_dgl_datasets import RAW_PARTS


DGL_NATIVE_DATASETS = {
    'CoraGraphDataset': ('cora', 'cora_v2'),
    'CiteseerGraphDataset': ('citeseer', 'citeseer'),
    'PubmedGraphDataset': ('pubmed', 'pubmed'),
}
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_RAW_BYTES = 128 * 1024 * 1024
READ_BYTES = 64 * 1024


class _CanonicalRedirect(HTTPRedirectHandler):
    def __init__(self, canonical_url):
        super().__init__()
        self.canonical_url = canonical_url

    def redirect_request(self, request, response, code, message, headers, newurl):
        if newurl != self.canonical_url:
            raise ValueError('Canonical reference redirect changed its fixed HTTPS URL')
        return super().redirect_request(request, response, code, message, headers, newurl)


def _open_canonical(url, timeout):
    opener = build_opener(_CanonicalRedirect(url))
    return opener.open(Request(url, headers={'User-Agent': 'AutoReproducer dataset-reference/1'}),
                       timeout=timeout)


def _download_reference(url, timeout):
    deadline = time.monotonic() + timeout
    # read1 performs one buffered/socket read, unlike read(n), which can keep
    # aggregating a slowly arriving body. The short socket timeout also bounds
    # an idle read; elapsed time is checked after every read, including EOF.
    with _open_canonical(url, min(timeout, 5)) as response:
        if response.geturl() != url or response.status != 200:
            raise ValueError('Canonical reference response lacks the fixed URL and complete HTTP 200 body')
        length = response.headers.get('Content-Length')
        if length is not None and (not re.fullmatch(r'[0-9]+', str(length))
                                   or not 0 < int(length) <= MAX_ARCHIVE_BYTES):
            raise ValueError('Canonical reference archive length is invalid or exceeds its bound')
        chunks, size = [], 0
        read = getattr(response, 'read1', None)
        if not callable(read):
            raise ValueError('Canonical reference response lacks a bounded single-read interface')
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('Canonical reference download exceeded its time bound')
            chunk = read(min(READ_BYTES, MAX_ARCHIVE_BYTES - size + 1))
            if time.monotonic() > deadline:
                raise TimeoutError('Canonical reference download exceeded its time bound')
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError('Canonical reference archive exceeds its byte bound')
            chunks.append(chunk)
        if not size or length is not None and size != int(length):
            raise ValueError('Canonical reference body is empty or differs from Content-Length')
        body = b''.join(chunks)
        receipt = {'source_url': url, 'final_url': response.geturl(),
                   'fetched_at': datetime.now(timezone.utc).isoformat(),
                   'archive_bytes': size, 'archive_sha256': hashlib.sha256(body).hexdigest()}
        for header, key in [('ETag', 'etag'), ('Last-Modified', 'last_modified')]:
            value = response.headers.get(header)
            if isinstance(value, str):
                receipt[key] = value[:512]
        return body, receipt


def _archive_manifest(body, native):
    expected = {f'ind.{native}.{part}' for part in RAW_PARTS}
    with ZipFile(BytesIO(body)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (len(entries) != len(expected) or len(set(names)) != len(names) or set(names) != expected
                or any(entry.orig_filename != entry.filename or '\x00' in entry.orig_filename
                       for entry in entries)):
            raise ValueError('Canonical ZIP must contain exactly the eight unique native raw members')
        if sum(entry.file_size for entry in entries) > MAX_RAW_BYTES:
            raise ValueError('Canonical ZIP raw members exceed the total byte bound')
        members = {}
        for entry in entries:
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if (entry.is_dir() or mode not in (0, stat.S_IFREG) or entry.flag_bits & 1
                    or entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
                    or not 0 < entry.file_size <= MAX_MEMBER_BYTES):
                raise ValueError('Canonical ZIP member is not a supported bounded regular raw file')
            checksum, size = hashlib.sha256(), 0
            with archive.open(entry) as stream:
                while True:
                    chunk = stream.read(min(READ_BYTES, MAX_MEMBER_BYTES - size + 1))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_MEMBER_BYTES:
                        raise ValueError('Canonical ZIP member exceeded its byte bound')
                    checksum.update(chunk)
            if size != entry.file_size:
                raise ValueError('Canonical ZIP member differs from its declared size')
            members[entry.filename] = {'bytes': size, 'sha256': checksum.hexdigest()}
        return members


def _observed_manifests(source):
    if (not isinstance(source, dict) or source.get('verified') is not True
            or source.get('kind') != 'author_runtime'):
        raise ValueError('Reference comparison needs an already verified author-runtime source')
    loader = source.get('loader')
    if not isinstance(loader, dict) or loader.get('framework') != 'dgl.data':
        raise ValueError('Reference comparison supports original DGL citation loaders only')
    identity = DGL_NATIVE_DATASETS.get(loader.get('dataset_class'))
    if identity is None:
        raise ValueError('Reference comparison has no canonical source for this DGL class')
    canonical, native = identity
    url = f'https://data.dgl.ai/dataset/{native}.zip'
    if (loader.get('framework_dataset_name') != native or loader.get('framework_version') != '1.1.2'
            or str(loader.get('dataset_name', '')).casefold() not in (canonical, native)):
        raise ValueError('DGL reference loader class, dataset or supported version is inconsistent')
    observations = source.get('observations')
    if not isinstance(observations, list) or not observations:
        raise ValueError('DGL reference comparison lacks complete observed raw files')
    expected = {f'ind.{native}.{part}' for part in RAW_PARTS}
    manifests = []
    for observation in observations:
        if (not isinstance(observation, dict) or observation.get('verified') is not True
                or observation.get('declaration') != loader
                or observation.get('framework_dataset_name') != native
                or observation.get('framework_version') != loader['framework_version']
                or observation.get('source_url') != url
                or str(observation.get('dataset_name', '')).casefold() not in (canonical, native)):
            raise ValueError('Observed DGL input is not bound to the verified canonical loader')
        raw_path, files = observation.get('raw_path'), observation.get('files')
        if not isinstance(raw_path, str) or not raw_path or not isinstance(files, dict):
            raise ValueError('Observed DGL input lacks its raw directory or file manifest')
        raw_prefix = raw_path.replace('\\', '/').rstrip('/') + '/'
        manifest = {}
        for path, record in files.items():
            if not isinstance(path, str) or not isinstance(record, dict):
                raise ValueError('Observed DGL file manifest is malformed')
            if record.get('role') != 'raw':
                continue
            normalized = path.replace('\\', '/')
            name = normalized.rsplit('/', 1)[-1]
            if (name not in expected or normalized != raw_prefix + name or name in manifest
                    or type(record.get('bytes')) is not int or record['bytes'] <= 0
                    or not isinstance(record.get('sha256'), str)
                    or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])):
                raise ValueError('Observed DGL raw file identity, size or checksum is incomplete')
            manifest[name] = {'bytes': record['bytes'], 'sha256': record['sha256']}
        if set(manifest) != expected:
            raise ValueError('Observed DGL input must cover all eight native raw files')
        manifests.append(manifest)
    return canonical, native, url, manifests


def audit_dgl_reference(source, *, timeout=30):
    """Audit one verified ``verify_runtime_datasets`` source, without changing it.

    ``verified`` and ``current_source_match`` describe only the independently
    downloaded current canonical raw bytes. Upstream published checksums and
    paper protocol verification deliberately remain separate and false.
    """
    result = {'version': 1, 'status': 'unavailable', 'verified': False, 'current_source_match': None,
              'basis': 'observed_raw_files_compared_with_current_canonical_zip',
              'published_checksum_verified': False, 'published_checksum_status': 'unknown',
              'paper_protocol_verified': False, 'paper_protocol_status': 'not_checked',
              'processed_inputs_verified': False, 'processed_inputs_status': 'not_compared',
              'canonical_reference': None,
              'comparisons': [], 'errors': [], 'limitations': [
                  'A current canonical-source snapshot is not an upstream published immutable checksum.',
                  'Processed graph tensors, masks, paper protocol and model metrics are not independently verified.',
              ]}
    stage = 'observation'
    try:
        canonical, native, url, manifests = _observed_manifests(source)
        result.update(dataset_name=canonical, framework_dataset_name=native,
                      framework_version='1.1.2', canonical_reference={'source_url': url})
        if type(timeout) not in (int, float) or not 0 < timeout <= 60:
            raise ValueError('Canonical reference timeout must be positive and at most 60 seconds')
        stage = 'download'
        body, receipt = _download_reference(url, timeout)
        result['canonical_reference'] = receipt
        stage = 'archive'
        members = _archive_manifest(body, native)
        receipt['members'] = members
        for index, observed in enumerate(manifests):
            for name in sorted(members):
                result['comparisons'].append({'observation_index': index, 'member': name,
                    'observed': observed[name], 'reference': members[name],
                    'matches': observed[name] == members[name]})
        matches = all(item['matches'] for item in result['comparisons'])
        result.update(status='verified' if matches else 'mismatch', verified=matches,
                      current_source_match=matches)
    except Exception as exc:
        result['errors'].append({'stage': stage, 'reason': str(exc)[:2000]})
    return result

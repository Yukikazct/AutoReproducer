"""Original DGL constructors retain their data paths and complete input evidence."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from src.runtime_dataset_evidence import audit_runtime_datasets, compile_runtime_dataset, discover_runtime_constructors, verify_runtime_datasets
from test_runtime_dataset_evidence import setup


def dgl_source(tmp_path, source):
    workspace, packet, item, citations = setup(tmp_path)
    path = workspace / 'data_loader.py'
    path.write_bytes(source.encode())
    packet['repository']['files'][path.name].update(text=source, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    citations['loader']['quote'] = source
    item.update(dataset_class='CoraGraphDataset', dataset_name='cora')
    return workspace, packet, item, citations


@pytest.mark.parametrize('imports,constructor', [
    ('from dgl.data import CoraGraphDataset', 'CoraGraphDataset'),
    ('from dgl import data as native', 'native.CoraGraphDataset'),
    ('import dgl.data as native', 'native.CoraGraphDataset'),
    ('import dgl', 'dgl.data.CoraGraphDataset'),
])
def test_original_dgl_constructor_imports_are_catalogued_and_source_bound(tmp_path, imports, constructor):
    source = f'{imports}\ndef get_dataset(path):\n    return {constructor}(raw_dir=path)\n'
    workspace, packet, item, citations = dgl_source(tmp_path, source)
    options = discover_runtime_constructors(packet['repository']['files'])
    assert any(row['dataset_class'] == 'CoraGraphDataset' and row['function'] == 'get_dataset' for row in options)
    compiled = compile_runtime_dataset(item, packet, workspace, citations)
    assert compiled['framework'] == 'dgl.data'
    assert compiled['framework_version'] == '1.1.2'
    assert compiled['framework_dataset_name'] == 'cora_v2'
    assert compiled['source_sha256'] == hashlib.sha256((workspace / 'data_loader.py').read_bytes()).hexdigest()
    assert compiled['constructor_quotes'] == [f'{constructor}(raw_dir=path)']
    (workspace / 'data_loader.py').write_bytes((source + '# changed\n').encode())
    with pytest.raises(ValueError, match='changed'):
        compile_runtime_dataset(item, packet, workspace, citations)


def test_dgl_loader_wrapper_is_not_the_original_constructor_scope(tmp_path):
    source = '''from dgl.data import CoraGraphDataset
def actual_loader(path):
    return CoraGraphDataset(raw_dir=path)
def get_dataset(path):
    return actual_loader(path)
'''
    workspace, packet, item, citations = dgl_source(tmp_path, source)
    with pytest.raises(ValueError, match='actual constructor scopes: actual_loader'):
        compile_runtime_dataset(item, packet, workspace, citations)


@pytest.mark.parametrize('name', ['citeseer', 'pubmed'])
def test_cora_class_cannot_establish_a_different_dgl_dataset(tmp_path, name):
    source = 'from dgl.data import CoraGraphDataset\n# citeseer pubmed\ndef get_dataset(path):\n    return CoraGraphDataset(raw_dir=path)\n'
    workspace, packet, item, citations = dgl_source(tmp_path, source)
    item['dataset_name'] = name
    with pytest.raises(ValueError, match='identity|dataset'):
        compile_runtime_dataset(item, packet, workspace, citations)


def test_author_execution_plan_retains_a_verified_original_dgl_constructor(tmp_path):
    from discovered_fixtures import make_experiment
    from src.discovered_repository_plan import build_evidence_packet
    from src.author_execution_plan import validate_author_execution_plan
    fixture = make_experiment(tmp_path)
    script = fixture['workspace'] / 'train.py'
    source = ('from dgl.data import CoraGraphDataset\n'
              'def original_loader(path):\n    return CoraGraphDataset(raw_dir=path)\n' + script.read_text('utf-8'))
    script.write_bytes(source.encode())
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
    next(c for c in fixture['proposal']['citations'] if c['id'] == 'script')['quote'] = source
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    proposal = deepcopy(fixture['proposal'])
    proposal.update(scope='author_code_execution', device='cpu', datasets=[{
        'kind': 'author_runtime', 'path': 'train.py', 'function': 'original_loader',
        'dataset_class': 'CoraGraphDataset', 'dataset_name': 'cora', 'citations': ['script']}])
    plan = validate_author_execution_plan(proposal, packet, fixture['workspace'])
    assert len(plan['datasets']) == 1 and plan['datasets'][0]['framework'] == 'dgl.data'
    assert plan['datasets'][0]['framework_dataset_name'] == 'cora_v2'
    assert plan['steps'][0]['argv'] == proposal['steps'][0]['argv']


@pytest.fixture(scope='module')
def native_cora(tmp_path_factory):
    deps, executable = os.environ.get('AUTOREPRO_TEST_DGL_DEPS'), os.environ.get('AUTOREPRO_TEST_DGL_PYTHON')
    if not deps or not executable:
        pytest.skip('Native DGL evidence requires dedicated reviewed DGL1.1.2 / CPython3.11 test runtime')
    deps, executable = Path(deps).resolve(), Path(executable).resolve()
    assert (deps / '.ready').is_file() and executable.is_file()
    assert not deps.is_relative_to(Path.cwd() / 'data')
    root = tmp_path_factory.mktemp('native_dgl_input')
    source = '''from dgl.data import CoraGraphDataset
def get_dataset(path):
    return CoraGraphDataset(raw_dir=path, verbose=False)
dataset = get_dataset(DATA_ROOT)
'''
    workspace, packet, item, citations = dgl_source(root, source)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    program = '''import hashlib,json,runpy,sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2]]
import torch,dgl
from dgl.data import CoraGraphDataset
from src.experiments.observe_datasets import DatasetObserver
workspace,data=Path(sys.argv[3]),Path(sys.argv[4])
declaration=json.loads(sys.argv[5])
baseline=CoraGraphDataset(raw_dir=data,verbose=False)
initial={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
original_init,original_getitem=CoraGraphDataset.__init__,CoraGraphDataset.__getitem__
def no_extra_item(self,index):
    raise AssertionError('observer must not fetch an extra item or execute author transforms')
CoraGraphDataset.__getitem__=no_extra_item
observer=DatasetObserver([declaration],workspace,lambda:0)
try:
    namespace=runpy.run_path(str(workspace/'data_loader.py'),init_globals={'DATA_ROOT':data})
    records=observer.finish()
finally:
    observer.close()
    CoraGraphDataset.__getitem__=original_getitem
assert CoraGraphDataset.__init__ is original_init
actual=namespace['dataset']
assert all(torch.equal(baseline._g.ndata[key],actual._g.ndata[key]) for key in baseline._g.ndata)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in initial.items())
print('DGL_RECEIPT='+json.dumps({'records':records,'framework_version':dgl.__version__,
    'original_source_unchanged':True,'framework_files_unchanged':True,'all_native_tensor_values_unchanged':True}))
'''
    result = subprocess.run([str(executable), '-I', '-X', 'utf8', '-c', program, str(deps), str(Path.cwd()),
        str(workspace), str(root / 'actual_cora'), json.dumps(declaration)],
        env={**os.environ, 'DGLBACKEND': 'pytorch'}, capture_output=True, text=True, encoding='utf-8', timeout=180)
    assert result.returncode == 0, result.stderr
    output = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('DGL_RECEIPT=')).split('=', 1)[1])
    assert output['framework_version'] == '1.1.2'
    assert (workspace / 'data_loader.py').read_bytes() == source.encode()
    return workspace, declaration, output


def test_actual_native_dgl_cora_binds_all_files_and_split_metadata(native_cora):
    workspace, declaration, output = native_cora
    record = output['records'][0]
    assert record['framework_dataset_name'] == 'cora_v2'
    assert record['num_nodes'] == 2708 and record['num_edges'] == 10556
    assert record['num_features'] == 1433 and record['num_classes'] == 7
    assert record['mask_counts'] == {'train': 140, 'validation': 500, 'test': 1000}
    assert len([f for f in record['files'].values() if f['role'] == 'raw']) == 8
    assert len([f for f in record['files'].values() if f['role'] == 'processed']) == 2
    assert record['source_url'] == 'https://data.dgl.ai/dataset/cora_v2.zip'
    assert verify_runtime_datasets({'datasets': [declaration]}, [{'native_datasets': output['records']}], workspace)[0]['verified']


@pytest.mark.parametrize('damage', ['missing_raw', 'native_identity', 'version', 'mask_count'])
def test_dgl_parent_rejects_incomplete_or_changed_native_evidence(native_cora, damage):
    workspace, declaration, output = native_cora
    records = deepcopy(output['records'])
    record = records[0]
    if damage == 'missing_raw':
        record['files'].pop(next(name for name, value in record['files'].items() if value['role'] == 'raw'))
    elif damage == 'native_identity':
        record['framework_dataset_name'] = 'pubmed'
    elif damage == 'version':
        record['framework_version'] = '0.0'
    else:
        record['mask_counts']['test'] = record['num_nodes'] + 1
    with pytest.raises(ValueError, match='DGL|dataset|Native'):
        verify_runtime_datasets({'datasets': [declaration]}, [{'native_datasets': records}], workspace)


def test_materialized_author_runtime_keeps_dgl_evidence_through_real_updates(native_cora):
    from src.author_execution_plan import materialize_author_execution, verify_author_execution
    from src.discovered_repository_runtime import CAPTURE_DIR
    workspace, _, output = native_cora
    root = output['records'][0]['root']
    source = f'''from dgl.data import CoraGraphDataset
import torch
dataset = CoraGraphDataset(raw_dir={root!r}, verbose=False)
data = dataset[0]
model = torch.nn.Linear(data.ndata['feat'].shape[1], dataset.num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(10):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(data.ndata['feat'])[data.ndata['train_mask']], data.ndata['label'][data.ndata['train_mask']])
    loss.backward()
    optimizer.step()
print('DGL_AUTHOR_COMPLETED')
'''
    path = workspace / 'native_train.py'
    path.write_bytes(source.encode())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = {'author_execution_only': True, 'repository': {'files': {
        path.name: {'text': source, 'sha256': sha}}}}
    declaration = compile_runtime_dataset({'kind': 'author_runtime', 'path': path.name,
        'function': '<module>', 'dataset_class': 'CoraGraphDataset', 'dataset_name': 'cora',
        'citations': ['loader']}, packet, workspace, {'loader': {
        'source': 'repository', 'path': path.name, 'quote': source}})
    step = {'id': 'train', 'kind': 'train', 'argv': ['python', path.name], 'cwd': '.',
        'entrypoint': {'kind': 'script', 'target': path.name, 'source_path': path.name},
        'timeout_s': None, 'required': True, 'depends_on': [], 'citations': []}
    plan = {'scope': 'author_code_execution', 'capture': {'device': 'cpu'}, 'steps': [step],
            'datasets': [declaration], 'requirements': {'compatibility': ['torch==2.5.1+cpu', 'dgl==1.1.2']}}
    materialized = materialize_author_execution(workspace, plan, 'a' * 64, {'files': {path.name: sha}})
    runtime = workspace / CAPTURE_DIR
    assert (runtime / 'observe_dgl_datasets.py').is_file()
    program = '''import runpy,sys
from pathlib import Path
runtime=Path(sys.argv[1])
sys.path[:0]=[str(runtime),sys.argv[2],sys.argv[3]]
sys.argv=[str(runtime/'observe_author.py'),'train']
runpy.run_path(str(runtime/'observe_author.py'),run_name='__main__')
'''
    result = subprocess.run([os.environ['AUTOREPRO_TEST_DGL_PYTHON'], '-I', '-X', 'utf8', '-c', program,
        str(runtime), os.environ['AUTOREPRO_TEST_DGL_DEPS'], str(Path.cwd())], cwd=workspace,
        env={**os.environ, 'DGLBACKEND': 'pytorch', 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'},
        capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'DGL_AUTHOR_COMPLETED' in result.stdout
    receipt = json.loads((runtime / 'author_train.json').read_text('utf-8'))
    assert receipt['optimizer_steps'] == 10 and receipt['natural_return'] is True
    assert receipt['native_datasets'][0]['loaded_before_training'] is True
    execution = {'success': True, 'steps': [{'id': 'train', 'executed': True, 'success': True,
                  'exit_code': 0, 'stdout': result.stdout, 'stderr': result.stderr}]}
    verified = verify_author_execution(plan, workspace, execution, 'a' * 64)
    assert audit_runtime_datasets(plan, verified['author_execution'], workspace)['verified'] is True
    assert verified['paper_metrics_verified'] is False and verified['is_reproduced'] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    assert materialized['files'][f'{CAPTURE_DIR}/observe_dgl_datasets.py']

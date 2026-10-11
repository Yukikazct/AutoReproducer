"""Graph packages need source imports, matching wheels and native health probes."""
import hashlib

import pytest

from discovered_fixtures import make_experiment
from src import discovered_repository_plan as planner
from src.discovered_repository_reproduction import environment_for_plan
from src.agents.code_executor import _reviewed_pip_requirement_sources


@pytest.mark.parametrize('device,suffix', [('cpu', 'cpu'), ('cuda', 'cu121')])
def test_actual_graph_imports_compile_package_aliases_and_runtime_dependencies(tmp_path, device, suffix):
    fixture = make_experiment(tmp_path)
    fixture['proposal']['requirements']['compatibility'][0] = f'torch==2.5.1+{suffix}'
    path = fixture['workspace'] / 'train.py'
    source = ('import torch_geometric\nfrom torch_sparse import coalesce\n'
              + ('import dgl\n' if device == 'cpu' else '') + path.read_text('utf-8'))
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item.get('path') == path.name)['quote'] = source
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    citations = planner._citations(fixture['proposal'], packet, fixture['workspace'])
    requirements = planner._requirements(fixture['proposal']['requirements'], citations, packet,
                                         device=device, step=fixture['proposal']['steps'][0])
    pins = set(requirements['compatibility'])
    assert {f'torch==2.5.1+{suffix}', 'torch-geometric==2.6.1',
            f'torch-sparse==0.6.18+pt25{suffix}', f'torch-scatter==2.1.2+pt25{suffix}',
            'psutil==6.1.0', 'fsspec==2024.10.0', 'jinja2==3.1.6', 'aiohttp==3.10.10'} <= pins
    assert ('dgl==1.1.2' in pins) == (device == 'cpu')
    environment = environment_for_plan({'capture': {'device': device}, 'requirements': requirements})
    assert f'--find-links https://data.pyg.org/whl/torch-2.5.0+{suffix}.html' in environment['requirements_txt']
    assert _reviewed_pip_requirement_sources(environment['requirements_txt'])
    assert any(item.get('dependency_of') == 'torch-sparse' and item['package'] == 'torch-scatter'
               for item in requirements['import_evidence'])


@pytest.mark.parametrize('url', [
    'https://data.pyg.org.attacker.example/whl/torch-2.5.0+cpu.html',
    'https://data.pyg.org/whl/torch-2.5.0+cpu.html?alternate=true',
    'https://data.pyg.org/whl/torch-2.4.0+cpu.html',
    'http://data.pyg.org/whl/torch-2.5.0+cpu.html',
])
def test_unreviewed_graph_wheel_sources_do_not_gain_public_fallback(url):
    assert not _reviewed_pip_requirement_sources('--find-links ' + url)

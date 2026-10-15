"""Coordinate local dependency users and cleanup across threads and processes."""
import os
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from src.safety.paths import is_link, workspace_path
from src.safety.files import remove_tree


LOCK_TIMEOUT = 60
# First-time preset setup can spend several minutes downloading native wheels.
# Other presets queue before their experiment clocks start instead of failing
# after the shorter legacy/optimization wait.
AUTO_PREPARATION_LOCK_TIMEOUT = 1800
_locks = weakref.WeakValueDictionary()
_registry_lock = threading.Lock()


class DependencyCacheBusy(RuntimeError):
    pass


def reset_managed_environment(root, directory):
    """Replace an incomplete package tree without traversing external links.

    Callers must hold the cache lock. pip --upgrade overlays files but leaves
    obsolete binaries and dist-info behind, so a failed install needs a clean
    target before its next attempt.
    """
    root, directory = Path(root).absolute(), Path(directory).absolute()
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise ValueError("dependency repair target escapes its managed cache") from exc
    if relative == Path("."):
        raise ValueError("dependency repair must not replace the cache root")
    checked = workspace_path(root, relative, "dependency repair target")
    if checked.exists():
        if not checked.is_dir():
            raise ValueError("dependency repair target must be a directory")
        for parent, directories, files in os.walk(checked, followlinks=False):
            if any(is_link(Path(parent) / name) for name in directories + files):
                raise ValueError("dependency repair target contains a symlink or junction")
        remove_tree(checked)
    checked.mkdir(parents=True, exist_ok=True)
    return checked


DEPENDENCY_HEALTH_PROBE = r'''
import importlib, importlib.metadata, json, os, re, sys
from pathlib import Path

target = Path(sys.argv[1]).resolve(strict=True)
requirements = json.loads(sys.argv[2])
targets = [Path(p).resolve(strict=True) for p in json.loads(sys.argv[3])] if len(sys.argv) > 3 else [target]
expected_versions = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
if any(req['module'] == 'dgl' for req in requirements):
    os.environ['DGLBACKEND'] = 'pytorch'
sys.path[:0] = [str(path) for path in targets]
# packaging can itself be one of the managed dependencies. Import the probe's
# helper only after selecting the cache, otherwise sys.modules keeps the host
# copy and a healthy managed packaging install is incorrectly rejected.
try:
    from packaging.specifiers import SpecifierSet
except ModuleNotFoundError as exc:
    if exc.name != 'packaging':
        raise
    # Fresh training venvs have pip, but need not install packaging separately.
    # Use pip's bundled parser without adding a dependency to the author plan.
    # Broken managed packaging modules must still fail the health check.
    from pip._vendor.packaging.specifiers import SpecifierSet
canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
distributions = {}
for path in targets:
    local = {}
    for dist in importlib.metadata.distributions(path=[str(path)]):
        name = canonical(dist.metadata.get('Name', ''))
        if name:
            if name in local:
                raise RuntimeError('duplicate distribution metadata: ' + name)
            local[name] = dist
    for name, dist in local.items():
        distributions.setdefault(name, dist)
for name, version in expected_versions.items():
    if name not in distributions or distributions[name].version != version:
        raise RuntimeError('resolved wheel version missing from managed environment: ' + name)
loaded = {}
for requirement in requirements:
    name = canonical(requirement['name'])
    if name not in distributions:
        raise RuntimeError('missing cached distribution: ' + name)
    dist = distributions[name]
    if requirement['specifier'] and not SpecifierSet(requirement['specifier']).contains(dist.version, prereleases=True):
        raise RuntimeError('cached version does not satisfy fixed requirement: ' + name + '==' + dist.version)
    module_name = requirement['module']
    module = importlib.import_module(module_name)
    locations = ([getattr(module, '__file__', None)] if getattr(module, '__file__', None)
                 else list(getattr(module, '__path__', ())))
    if not locations or any(not any(Path(location).resolve().is_relative_to(path) for path in targets) for location in locations):
        raise RuntimeError('package imported outside the managed cache: ' + module_name)
    loaded[module_name] = module

# Importing a package alone can miss broken compiled extensions or incompatible
# wheel pairs. Execute small real native operations before publishing .ready.
if 'numpy' in loaded:
    np = loaded['numpy']
    assert float((np.ones((2, 2)) @ np.ones((2, 2)))[0, 0]) == 2.0
if 'torch' in loaded:
    torch = loaded['torch']
    assert (torch.ones((2, 2)) @ torch.ones((2, 2))).sum().item() == 8.0
    if 'numpy' in loaded:
        assert torch.from_numpy(loaded['numpy'].ones(1)).sum().item() == 1.0
if 'scipy' in loaded:
    from scipy.linalg import solve
    assert float(solve([[1., 0.], [0., 1.]], [1., 2.])[1]) == 2.0
if 'pandas' in loaded:
    assert loaded['pandas'].DataFrame({'x': [1, 2]}).x.sum() == 3
if 'sklearn' in loaded:
    from sklearn.utils import murmurhash3_32
    assert isinstance(murmurhash3_32('dependency-health'), int)
if 'PIL' in loaded:
    from PIL import Image
    assert Image.new('RGB', (2, 2)).size == (2, 2)
if 'matplotlib' in loaded:
    loaded['matplotlib'].use('Agg')
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(1, 1)); figure.subplots().plot([0, 1]); FigureCanvasAgg(figure).draw()
if 'torchvision' in loaded:
    torch = loaded['torch']
    result = loaded['torchvision'].ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), .5)
    assert result.tolist() == [0]
if 'torch_scatter' in loaded:
    torch = loaded['torch']
    assert loaded['torch_scatter'].scatter_add(torch.tensor([1., 2.]), torch.tensor([0, 0])).tolist() == [3.]
if 'torch_sparse' in loaded:
    torch = loaded['torch']
    edge = torch.tensor([[0, 0], [1, 1]])
    indices, values = loaded['torch_sparse'].coalesce(edge, torch.tensor([1., 2.]), 2, 2)
    assert indices.tolist() == [[0], [1]] and values.tolist() == [3.]
if 'torch_geometric' in loaded:
    from torch_geometric.nn import APPNP
    torch = loaded['torch']
    output = APPNP(2, .1)(torch.tensor([[1.], [2.]]), torch.tensor([[0, 1], [1, 0]]))
    assert output.shape == (2, 1) and torch.isfinite(output).all()
if 'dgl' in loaded:
    dgl, torch = loaded['dgl'], loaded['torch']
    graph = dgl.graph(([0, 1], [1, 0]))
    graph.ndata['x'] = torch.tensor([[1.], [2.]])
    graph.update_all(dgl.function.copy_u('x', 'm'), dgl.function.sum('m', 'out'))
    assert graph.ndata['out'].tolist() == [[2.], [1.]]
print(json.dumps({'versions': {name: distributions[canonical(req['name'])].version
                              for name, req in [(r['module'], r) for r in requirements]}}))
'''


@contextmanager
def cache_guard(root, *, cleanup=False, timeout=None):
    """Serialize use of one cache root; cleanup never waits for active users.

    A shared lock instance is reentrant within the owning thread, so dependency
    preparation/self-healing can also safely be called on their own. Cleanup
    explicitly rejects even a same-thread active user.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = workspace_path(root, ".autorepro-cache.lock", "dependency lock")
    key = (os.getpid(), os.path.normcase(str(lock_path)))
    with _registry_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = FileLock(str(lock_path))
            _locks[key] = lock
    wait = 0 if cleanup else (LOCK_TIMEOUT if timeout is None else timeout)
    if cleanup and lock.is_locked:
        raise DependencyCacheBusy("依赖缓存正在使用，本次清理已跳过")
    try:
        lock.acquire(timeout=wait)
    except Timeout as exc:
        message = ("依赖缓存正在使用，本次清理已跳过" if cleanup else
                   f"依赖缓存正在使用，等待 {wait:g} 秒超时，请稍后重试")
        raise DependencyCacheBusy(message) from exc
    try:
        yield root
    finally:
        lock.release()

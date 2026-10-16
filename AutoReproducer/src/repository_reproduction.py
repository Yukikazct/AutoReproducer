"""A fixed DLinear experiment from an authentic Git snapshot to final test metrics.

Preparation is separate from execution. Shared source/data caches are read-only
inputs; every invocation exports a fresh commit tree, without .git or old results.
"""
import csv
import os
import hashlib
import io
import json
import math
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
from urllib.parse import urljoin, urlsplit
import uuid
from pathlib import Path
from filelock import FileLock

from src.repository_profiles import get_profile
from src.repository_adapters import get_adapter
from src.repository_archive import validated_archive
from src.execution_plan import build_plan, plan_step_ids
from src.safety.paths import workspace_path
from src.preset_downloads import atomic_cache_bytes, download_bytes
from src.environment_lifecycle import bind_environment_run, release_run_environment


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_digest(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _git(repo, *args, timeout=30, binary=False):
    command = ["git", "-C", str(repo)]
    if args and args[0] == "archive":
        # Export the same committed source bytes on every host. User checkout
        # defaults must not alter the immutable archive or its evidence hashes.
        command += ["-c", "core.autocrlf=false", "-c", "core.eol=lf"]
    result = subprocess.run([*command, *args], capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip()[-1000:])
    return result.stdout if binary else result.stdout.decode("utf-8", "replace").strip()


def _canonical_url(url):
    return url.rstrip("/").removesuffix(".git").lower()


def _verified_repository_archive(source, url, revision):
    origin = _git(source, "remote", "get-url", "origin")
    commit = _git(source, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if _canonical_url(origin) != _canonical_url(url) or commit != revision:
        raise RuntimeError("仓库缓存来源或固定版本校验失败")
    # rev-parse can succeed even when a referenced tree/blob is missing. Read
    # the committed archive before selecting a cache so online preparation can
    # recover broken objects rather than failing later in a new workspace.
    return _git(source, "archive", "--format=tar", revision, binary=True)


def _discard_temporary_repository(source, parent):
    source, parent = Path(source), Path(parent).resolve()
    resolved = source.resolve()
    if resolved.parent != parent or not source.name.startswith(".fetch-"):
        raise ValueError("临时仓库清理路径超出固定缓存目录")

    def clear_read_only(operation, filename, error_info):
        error = error_info[1]
        path = Path(filename)
        if not isinstance(error, PermissionError) or not path.resolve().is_relative_to(resolved):
            raise error
        # Git marks pack/index files read-only on Windows. Only the checked
        # temporary tree is writable here; cached author trees stay untouched.
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
        operation(filename)

    if source.exists():
        shutil.rmtree(source, onerror=clear_read_only)


def _transient_git_error(error):
    if isinstance(error, subprocess.TimeoutExpired):
        return True
    return bool(re.search(
        r"could not resolve host|failed to connect|connection (?:reset|timed out)|"
        r"operation timed out|remote end hung up|early eof|rpc failed|"
        r"http/?[12](?:\.[01])? (?:408|429|5\d\d)|returned error:\s*(?:408|429|5\d\d)",
        str(error), re.I))


def _fetch_repository(cache, url, revision):
    """Fetch exact author objects, then atomically publish a verified cache."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        source = cache.parent / f".fetch-{uuid.uuid4().hex}"
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--single-branch", url, str(source)],
                capture_output=True, timeout=300)
            if result.returncode:
                raise RuntimeError(result.stderr.decode("utf-8", "replace")[-1000:])
            _git(source, "fetch", "--depth", "1", "origin", revision, timeout=120)
            _git(source, "checkout", "--detach", revision)
            if _git(source, "rev-parse", "HEAD") != revision:
                raise RuntimeError("固定仓库版本校验失败，停止执行")
            archive = _verified_repository_archive(source, url, revision)
            break
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            _discard_temporary_repository(source, cache.parent)
            if attempt == 3 or not _transient_git_error(error):
                raise RuntimeError(f"固定仓库下载失败；已尝试 {attempt} 次，来源或固定版本未通过核验") from None
            time.sleep(2 ** (attempt - 1))
    with FileLock(str(cache.parent / f".{revision}.prepare.lock"), timeout=30):
        if cache.exists():
            try:
                current_archive = _verified_repository_archive(cache, url, revision)
            except (OSError, subprocess.SubprocessError, RuntimeError):
                # Preserve a corrupt cache for diagnosis; never rewrite Git
                # objects in place while another preparation may use them.
                checked_cache = workspace_path(cache.parent, cache.name, "repository cache", must_exist=True)
                quarantine = workspace_path(cache.parent, f".invalid-{revision}-{uuid.uuid4().hex}", "repository cache")
                checked_cache.rename(quarantine)
            else:
                _discard_temporary_repository(source, cache.parent)
                return cache, current_archive
        try:
            source.rename(cache)
            source = cache
        except OSError:
            pass  # The verified unique tree remains usable if publication fails.
    return source, archive


def _submodule_url(parent_url, value):
    """Resolve Git-relative URLs without changing a submodule's pinned commit."""
    if not value:
        raise ValueError('子模块缺少固定来源 URL')
    if '://' not in parent_url and Path(parent_url).is_absolute():
        return str((Path(parent_url) / value).resolve()) if value.startswith('.') else value
    if value.startswith(('./', '../')):
        value = urljoin(parent_url.rstrip('/') + '/', value)
    if value.startswith('git@github.com:'):
        value = 'https://github.com/' + value.split(':', 1)[1]
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('子模块需要公开 HTTPS 来源')
    return value


def materialize_submodules(data_root, snapshot, *, offline=False, _ancestors=()):
    """Export each committed gitlink, retaining hashes and nested source provenance."""
    identity = (snapshot['url'], snapshot['resolved_sha'])
    if identity in _ancestors or len(_ancestors) >= 8:
        raise ValueError('子模块引用循环或嵌套超过八层')
    dest, source = Path(snapshot['path']), Path(snapshot['cache_path'])
    gitlinks = []
    for record in _git(source, 'ls-tree', '-r', '-z', snapshot['resolved_sha'], binary=True).split(b'\0'):
        if not record:
            continue
        header, path = record.split(b'\t', 1)
        mode, kind, revision = header.decode('ascii').split()
        if mode == '160000' and kind == 'commit':
            gitlinks.append((path.decode('utf-8'), revision))
    if len(gitlinks) > 64:
        raise ValueError('单个仓库的子模块数量超过六十四个')
    modules = dest / '.gitmodules'
    configured = {}
    if gitlinks:
        if not modules.is_file():
            raise ValueError('固定 gitlink 缺少 .gitmodules 来源声明')
        raw = _git(source, 'config', '--file', str(modules.resolve()), '--null', '--get-regexp',
                   r'^submodule\..*\.(path|url)$', binary=True)
        for item in raw.split(b'\0'):
            if item:
                key, value = item.decode('utf-8').split('\n', 1)
                name, field = key.rsplit('.', 1)
                configured.setdefault(name, {})[field] = value
    by_path = {}
    for item in configured.values():
        if 'path' in item:
            if item['path'] in by_path:
                raise ValueError('子模块路径声明重复')
            by_path[item['path']] = item.get('url', '')
    snapshots = []
    for name, revision in gitlinks:
        target = workspace_path(dest, name, 'submodule', forbid_git=True)
        child_url = _submodule_url(snapshot['url'], by_path.get(name, ''))
        if target.exists():
            target.rmdir()  # Only a Git archive's empty gitlink placeholder is allowed.
        child = export_repository(data_root, {'repository': {'url': child_url, 'revision': revision}},
            target, offline=offline, required_files=[], include_submodules=False)
        materialize_submodules(data_root, child, offline=offline, _ancestors=(*_ancestors, identity))
        snapshots.append({'mount_path': name, **child})
        for path, sha in child['files'].items():
            qualified = f'{name}/{path}'
            if qualified in snapshot['files']:
                raise ValueError('子模块不能覆盖父仓库文件')
            snapshot['files'][qualified] = sha
        for alias in child.get('file_aliases', []):
            snapshot.setdefault('file_aliases', []).append({
                **alias, 'path': f"{name}/{alias['path']}",
                'resolved_target': f"{name}/{alias['resolved_target']}"})
    snapshot['submodules'] = snapshots
    snapshot['submodules_pending'] = False
    return snapshot


def resolve_repository_revision(data_root, url, revision=""):
    """Resolve a legacy SHA/ref request to a commit without falling back to HEAD."""
    requested = revision.strip() or "HEAD"
    if re.fullmatch(r"[a-fA-F0-9]{40}", requested):
        return requested.lower()
    if requested.startswith("-") or "\x00" in requested:
        raise ValueError("固定仓库版本无效")
    if re.fullmatch(r"[a-fA-F0-9]{7,39}", requested):
        # A server need not accept abbreviated commits. Failure is explicit;
        # FETCH_HEAD is used only after this exact request succeeds.
        parent = workspace_path(data_root, "repository_cache/resolution", "revision resolution")
        parent.mkdir(parents=True, exist_ok=True)
        source = parent / f".fetch-{uuid.uuid4().hex}"
        source.mkdir()
        try:
            _git(source, "init")
            _git(source, "fetch", "--depth", "1", url, requested, timeout=120)
            resolved = _git(source, "rev-parse", "--verify", "FETCH_HEAD^{commit}")
        finally:
            _discard_temporary_repository(source, parent)
    else:
        names = {requested, f"refs/heads/{requested}", f"refs/tags/{requested}"}
        patterns = sorted(names | {name + "^{}" for name in names})
        result = subprocess.run(["git", "ls-remote", "--exit-code", "--", url, *patterns],
                                capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError("无法解析请求的固定仓库版本，停止获取源码")
        refs = {}
        for line in result.stdout.decode("utf-8", "replace").splitlines():
            sha, separator, name = line.partition("\t")
            if separator:
                refs[name] = sha
        candidates = {refs.get(name + "^{}", refs[name]) for name in names if name in refs}
        if len(candidates) != 1:
            raise RuntimeError("请求的仓库版本不存在或存在歧义，停止获取源码")
        resolved = candidates.pop()
    if not re.fullmatch(r"[a-f0-9]{40}", resolved):
        raise RuntimeError("无法确定完整仓库 commit，停止获取源码")
    return resolved


def export_repository(data_root, profile, destination, *, offline=False, required_files=None,
                      include_submodules=True):
    """Export the requested commit, not potentially modified cached working files."""
    root, dest = Path(data_root), Path(destination)
    request = profile["repository"]
    url, revision = request["url"], request["revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("仓库复现需要完整固定commit SHA")
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache = root / "repository_cache" / key / revision
    # Migrate existing caches without rewriting or changing their source markers.
    candidates = [cache, root / "repos" / "dlinear_etth1_cpu_smoke" / "main",
                  root / "repos" / "fb52f754d198" / "main"]
    source = None
    for candidate in candidates:
        if not (candidate / ".git").exists():
            continue
        try:
            archive = _verified_repository_archive(candidate, url, revision)
            source = candidate
            break
        except (OSError, subprocess.SubprocessError, RuntimeError):
            continue
    acquisition = "cached" if source is not None else "cloned"
    if source is None:
        if offline:
            raise RuntimeError("离线模式中没有来源与固定版本均可核验的仓库缓存")
        source, archive = _fetch_repository(cache, url, revision)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        # Validate the entire archive before creating any extracted files.
        members, file_sources, file_aliases = validated_archive(tar, dest)
        dest.mkdir(parents=True, exist_ok=False)
        for name, member in members.items():
            # Recheck immediately before writing; earlier entries may exist now.
            target = workspace_path(dest, name, "archive", forbid_git=True)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = tar.extractfile(members[file_sources[name]]).read()
                target.write_bytes(payload)
                files[name] = hashlib.sha256(payload).hexdigest()
        for alias in file_aliases:
            alias['target_sha256'] = files[alias['path']]
    snapshot = {"url": url, "revision": revision, "resolved_sha": revision,
            "acquisition": acquisition,
            "path": str(dest.resolve()), "cache_path": str(source.resolve()),
             "archive_sha256": hashlib.sha256(archive).hexdigest(), "files": files,
             "file_aliases": file_aliases,
             "submodules_pending": (dest / '.gitmodules').is_file(), "submodules": []}
    if include_submodules and snapshot['submodules_pending']:
        materialize_submodules(root, snapshot, offline=offline)
    for path in (get_adapter(profile).required_files(profile) if required_files is None else required_files):
        if not workspace_path(dest, path, "required file", forbid_git=True).is_file():
            raise RuntimeError(f"固定仓库缺少必需入口: {path}")
    return snapshot


def prepare_dataset(data_root, profile, workspace, *, offline=False):
    root = Path(data_root)
    spec = profile["dataset"]
    cache = root / "dataset_cache" / spec["name"] / spec["sha256"] / "ETTh1.csv"
    legacy = root / "datasets"
    candidates = [cache, legacy / "dlinear_etth1_cpu_smoke" / "real" /
                  f"ETTh1-{spec['revision']}" / "ETTh1.csv",
                  legacy / "fb52f754d198" / "real" / f"ETTh1-{spec['revision']}" / "ETTh1.csv"]
    source = next((p for p in candidates if p.is_file() and
                   p.stat().st_size == spec["bytes"] and sha256_file(p) == spec["sha256"]), None)
    if source is None:
        if offline:
            raise RuntimeError("离线模式中缺少真实ETTh1数据或数据校验和不符")
        content = download_bytes(spec["url"], max_bytes=spec["bytes"], timeout_s=60)
        if len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise RuntimeError("真实ETTh1下载内容校验失败；不会替换为合成数据")
        atomic_cache_bytes(cache, content)
        source = cache
    with source.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = sum(1 for _ in reader)
    if header != ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"] or rows != spec["rows"]:
        raise RuntimeError("真实ETTh1列名或行数与冻结数据定义不一致")
    target = Path(workspace) / spec["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {**spec, "path": str(target.resolve()), "cache_path": str(source.resolve()),
            "verified": True, "rows": rows}


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf(?:inity)?)"
_FINAL_METRICS = re.compile(rf"^mse:\s*({_NUMBER})\s*,\s*mae:\s*({_NUMBER})\s*$", re.I | re.M)


def dlinear_metric_records(execution, spec_hash):
    """Only the author's final test block is a measurement; train loss is ignored."""
    final = execution.get("final") or {}
    if not execution_succeeded(execution):
        raise ValueError("必需执行步骤未成功，不能接受指标")
    stdout = final.get("stdout") or ""
    marker = ">>>>>>>testing :"
    if stdout.count(marker) != 1:
        raise ValueError("需要本次单次官方test阶段日志，不能复用历史/训练指标")
    matches = list(_FINAL_METRICS.finditer(stdout.split(marker, 1)[1]))
    if len(matches) != 1:
        raise ValueError("官方test阶段缺少唯一的最终MSE/MAE输出")
    values = [float(v) for v in matches[0].groups()]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("最终MSE/MAE必须是有限非负数")
    return [{"name": name, "value": value, "unit": "scalar", "split": "test",
             "stage": "eval", "seed": 2021, "source": final.get("stdout_path", "train_and_eval.stdout.log"),
             "spec_sha256": spec_hash}
            for name, value in zip(["mse", "mae"], values)]


def execution_succeeded(execution):
    final = execution.get("final") or {}
    steps = execution.get("steps") or execution.get("stages") or []
    # A blocked step is a failed verdict even if every attempt that ran succeeded;
    # metrics from a partially executed plan must never reach the paper comparison.
    return bool(execution.get("success") and not execution.get("skipped")
                and final.get("exit_code") == 0
                and all(step.get("success") is True and step.get("exit_code") == 0
                        and not step.get("timed_out") and not step.get("cancelled")
                        for step in steps if step.get("required") is not False))


def recompute_dlinear_metrics(execution, workspace, dataset, records):
    """Rebuild test labels from the checked CSV, independent of author eval code."""
    predictions = list((Path(workspace) / "results").glob("*/pred.npy"))
    if len(predictions) != 1:
        return {"pass": False, "reason": "缺少本次唯一预测数组"}
    artifact_dir = Path(workspace).parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    script = r'''
import json,sys,re,numpy as np
from pathlib import Path
prediction=np.load(sys.argv[1],allow_pickle=False)
raw=np.genfromtxt(sys.argv[2],delimiter=',',skip_header=1,usecols=range(1,8))
train=raw[:8640]
scaled=(raw-train.mean(axis=0))/train.std(axis=0)
truth=np.lib.stride_tricks.sliding_window_view(scaled[11520:14400],96,axis=0).transpose(0,2,1).astype(np.float32)
if prediction.shape != truth.shape or prediction.shape != (2785,96,7):
    raise ValueError('prediction/test label shape mismatch')
if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
    raise ValueError('non-finite prediction/test label')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[3])
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(range(1,97),truth[0,:,-1],label='Ground truth')
ax.plot(range(1,97),prediction[0,:,-1],label='DLinear prediction')
ax.set(xlabel='Forecast horizon (hours)',ylabel='Standardized oil temperature',title='ETTh1: first test forecast, 336 -> 96')
ax.legend();fig.tight_layout();fig.savefig(out/'test_forecast.png',dpi=160);plt.close(fig)
log=Path(sys.argv[4]).read_text(encoding='utf-8')
rows=re.findall(r'Epoch: (\d+), Steps: \d+ \| Train Loss: ([\d.eE+-]+) Vali Loss: ([\d.eE+-]+)',log)
if rows:
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot([int(r[0]) for r in rows],[float(r[1]) for r in rows],marker='o',label='Train MSE')
    ax.plot([int(r[0]) for r in rows],[float(r[2]) for r in rows],marker='o',label='Validation MSE')
    ax.set(xlabel='Epoch',ylabel='Loss',title='DLinear: actual training and validation losses')
    ax.legend();fig.tight_layout();fig.savefig(out/'training_losses.png',dpi=160);plt.close(fig)
print(json.dumps({'mse':float(np.mean((prediction-truth)**2)), 'mae':float(np.mean(np.abs(prediction-truth))),
                  'shape':list(prediction.shape),'numpy':np.__version__},allow_nan=False))
'''
    env = os.environ.copy()
    env.pop("LLM_API_KEY", None)
    environment = execution.get('environment') or {}
    deps = environment.get('dependency_paths') or [environment.get('dependencies_path', '')]
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in deps if path)
    try:
        proc = subprocess.run([sys.executable, "-c", script, str(predictions[0]), dataset["path"],
                               str(artifact_dir), execution["final"]["stdout_path"]],
                              capture_output=True, encoding="utf-8", errors="replace", env=env,
                              timeout=60)
        if proc.returncode:
            raise ValueError(proc.stderr[-1000:])
        actual = json.loads(proc.stdout)
        printed = {r["name"]: r["value"] for r in records}
        passed = (actual["numpy"] == "1.26.4" and all(
            math.isfinite(actual[name]) and abs(actual[name] - printed[name]) <= 1e-6
            for name in ["mse", "mae"]))
        return {"pass": passed, "metrics": actual, "source": str(predictions[0]),
                "artifact_dir": str(artifact_dir),
                "dataset_sha256": dataset["sha256"],
                "reason": "独立重建真实test标签并复算MSE/MAE，与作者日志在1e-6内一致" if passed else "独立复算与作者指标/锁定环境不一致"}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        return {"pass": False, "reason": str(exc)}


def validate_repository(profile, execution, records, error=""):
    paper = profile["paper"]["metrics"]
    actual = {r["name"]: r["value"] for r in records}
    numerical = None
    required = profile["paper"]["required_metrics"]
    final = execution.get("final") or {}
    if not error and (set(actual) != set(required) or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in actual.values())):
        error = "缺少必需最终指标或指标无效"
    if error or not execution_succeeded(execution):
        status, match, level = "execution_failed", False, "failed"
        reason = error or str((execution.get("final") or {}).get("stderr") or "仓库执行失败")[-1000:]
    elif profile["validation"]["level"] == "smoke":
        status, match, level = "smoke_passed", None, "smoke_passed"
        reason = "真实官方仓库与ETTh1数据完成1轮训练和test评估；这是冒烟通过，不能作为论文数值复现。"
    else:
        tolerance = profile["validation"]["relative_tolerance"]
        numerical = all(name in actual and abs(actual[name] - value) / abs(value) <= tolerance
                        for name, value in paper.items())
        protocol = execution.get("protocol_verification") or {}
        recomputed = execution.get("independent_metrics") or {}
        verified = (protocol.get("pass") is True and recomputed.get("pass") is True)
        match = numerical if verified else None
        status = ("reproduced" if numerical else "not_reproduced") if verified else "inconclusive"
        level = "reproduced" if match else "experiment_completed"
        reason = (f"完整作者实验已完成；与Table 2所有指标的相对误差{'均在' if numerical else '未全部落在'}"
                  f"{tolerance:.0%}以内。" +
                  ("固定版本、真实数据、训练/早停/测试协议和独立指标复算均通过；结论仅适用于本项论文实验。"
                   if verified else "完整运行协议或独立指标复算尚未通过，结论待核验。"))
    return {"status": status, "is_reproduced": match, "reason": reason,
            "result_level": level, "execution_status": level, "confidence": 0.0,
            "metrics_comparison": {"paper": paper, "actual": actual},
            "metric_records": records, "llm_calls": 0,
            "scope": "selected_paper_experiment",
            "validation": {"match": match, "analysis": reason, "verdict_source": "deterministic",
                           "differences": [f"{name}: 论文 {value}，实际 {actual[name]}，相对差异 {abs(actual[name]-value)/abs(value):.2%}"
                                           for name, value in paper.items() if name in actual],
                           "metrics_within_tolerance": numerical,
                           "relative_tolerance": profile["validation"]["relative_tolerance"]}}


class RepositoryReproduction:
    def __init__(self, data_root, logger, use_docker=False, runner=None, llm=None):
        self.root = Path(data_root)
        self.logger = logger
        self.use_docker = use_docker
        self.runner = runner
        self.llm = llm

    @release_run_environment
    def run(self, input_data, on_event=None):
        if get_profile(input_data["experiment_profile"]).get("adapter_id") in {"siren", "neural_ode", "rezero"}:
            from src.method_reproduction import MethodReproduction
            return MethodReproduction(self.root, self.logger, self.runner, self.llm).run(
                {**input_data, "use_docker": self.use_docker}, on_event)
        if input_data.get("prepare_environment"):
            raise ValueError("本预设不支持独立准备完整环境；请选择方法实验预设或仅准备源码")
        from src.agents.report_generator import ReportGeneratorAgent
        from src.repository_runner import RepositoryRunner
        data = {"verifications": [], "fix_records": [], "total_llm_calls": 0}
        state, error = "INIT", None
        initial_calls = self.llm.get_call_count() if self.llm else 0
        counted_calls = 0
        active_phase_id, active_agent = "", ""
        public_packet = None
        run_dir = self.root / "runs" / f"repository_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True)
        bind_environment_run(run_dir)
        data["run_dir"] = str(run_dir.resolve())
        data["report_path"] = str((run_dir / "report.md").resolve())
        if input_data.get("title_resolution"):
            data["title_resolution"] = dict(input_data["title_resolution"])
        if input_data.get("pdf_resolution"):
            data["pdf_resolution"] = dict(input_data["pdf_resolution"])

        def emit(event):
            if on_event:
                on_event(event)

        def sync_calls():
            nonlocal counted_calls
            actual_calls = max(0, (self.llm.get_call_count() if self.llm else 0) - initial_calls)
            self.logger.add_llm_calls(actual_calls - counted_calls)
            counted_calls = actual_calls
            data["total_llm_calls"] = actual_calls

        def phase(name, agent, action, phase_id):
            nonlocal state, active_phase_id, active_agent
            state = name
            active_phase_id, active_agent = phase_id, agent
            emit({"type": "state", "state": name, "agent": agent,
                  "phase_id": phase_id, "status": "running"})
            self.logger.begin_plan(name)
            self.logger.log(agent, name, "RUNNING", f"进入仓库复现阶段: {name}")
            try:
                value = action()
                ok = not isinstance(value, dict) or (
                    value.get("success") is not False and value.get("result_level") != "failed")
                self.logger.log(agent, name, "SUCCESS" if ok else "ERROR", f"仓库复现阶段结束: {name}")
                title = data.get("paper_title", "")
                if not title and isinstance(value, dict):
                    title = (value.get("paper") or {}).get("title", "")
                self.logger.log_experiment(name, "固定仓库实验", outputs={"title": title},
                                           result={"success": ok, "run_dir": str(run_dir)})
                details = {}
                if isinstance(value, dict):
                    outcome = value.get("outcome", value.get("status"))
                    if outcome is not None:
                        details["outcome"] = outcome
                    if value.get("reason"):
                        details["reason"] = value["reason"]
                emit({"type": "state", "state": name, "agent": agent, "phase_id": phase_id,
                      "status": "success" if ok else "error", **details})
                return value
            except Exception as exc:
                emit({"type": "state", "state": name, "agent": agent,
                      "phase_id": phase_id, "status": "error", "reason": str(exc)})
                raise
            finally:
                self.logger.end_plan(name)

        try:
            profile = phase("READ_PAPER", "PaperReader", lambda: get_profile(input_data["experiment_profile"]),
                            "select_experiment")
            from copy import deepcopy
            profile = deepcopy(profile)
            profile['steps'] = [{**step, 'depends_on': [], 'timeout_s': None}
                                for step in profile['steps'] if step.get('kind') != 'check']
            profile['validation'] = {**profile['validation'], 'scope': 'author_code_execution', 'reviews_removed': True}
            adapter = get_adapter(profile)
            data.update(paper_title=profile["paper"]["title"], paper_info=profile["paper"],
                        experiment_spec=profile, spec_sha256=spec_digest(profile))
            prepare_only = bool(input_data.get("prepare_only"))
            stages = [
                ("select_experiment", "PaperReader", "选择运行命令", "读取作者实验参数", True),
                ("prepare_repository", "ResourceFinder", "准备源码与数据", "取得作者代码及输入数据", True),
                ("prepare_environment_plan", "EnvBuilder", "准备依赖计划", "选择运行依赖", True),
                ("prepare_dependencies", "EnvBuilder", "安装运行依赖", "准备所需环境", not prepare_only),
                ("execute_repository", "CodeExecutor", "代码执行", "运行原作者命令", not prepare_only),
                ("generate_report", "ReportGenerator", "生成报告", "汇总运行输出和退出状态", True),
            ]
            emit({"type": "pipeline_plan", "pipeline": "repository", "stages": [
                {"id": identifier, "agent": agent, "title": title, "description": description,
                 "status": "waiting" if enabled else "skipped"}
                for identifier, agent, title, description, enabled in stages]})
            write_json(run_dir / "experiment_spec.json", {"spec": profile, "sha256": data["spec_sha256"]})
            offline = bool(input_data.get("offline"))

            def prepare():
                self.logger.log("ResourceFinder", "prepare_repository", "RUNNING", "正在获取作者仓库的固定版本：" + profile["repository"]["url"])
                snapshot = export_repository(self.root, profile, run_dir / "repo", offline=offline)
                data["repository"] = snapshot
                write_json(run_dir / "repository.json", snapshot)
                self.logger.log("ResourceFinder", "prepare_repository", "RUNNING", "源码已就绪；正在准备数据集：" + profile["dataset"].get("name", ""))
                dataset = adapter.prepare_dataset(self.root, profile, run_dir / "repo", offline=offline)
                data["dataset_provenance"] = dataset
                write_json(run_dir / "dataset.json", dataset)
                data["resources"] = {"code_repo_url": snapshot["url"], "dataset_url": dataset["url"],
                                     "confidence": 1.0}
                return snapshot

            snapshot = phase("FIND_RESOURCES", "ResourceFinder", prepare, "prepare_repository")

            def plan_environment():
                plan = build_plan(mode="repository", profile=profile["id"],
                                  spec_sha256=data["spec_sha256"], workspace=snapshot["path"],
                                  repository={"url": snapshot.get("url", ""),
                                              "revision": snapshot.get("resolved_sha")
                                              or snapshot.get("revision", "")},
                                  dataset={"name": profile["dataset"]["name"],
                                           "sha256": profile["dataset"]["sha256"]},
                                  steps=profile["steps"])
                data["execution_plan"] = plan
                write_json(run_dir / "execution_plan.json", plan)
                # The plan on disk must describe exactly the steps that run; a write
                # that failed or a rewritten file cannot silently change the verdict.
                stored = read_json(run_dir / "execution_plan.json")
                if plan_step_ids(stored.get("steps")) != plan_step_ids(plan["steps"]) or \
                        stored.get("spec_sha256") != plan["spec_sha256"]:
                    raise RuntimeError("落盘执行计划与本次执行步骤不一致")
                environment = {**profile["environment"], "dependency_health_check": False,
                               "auto_prepare": True, "offline": offline, "python_version": platform.python_version(),
                               "platform": platform.platform(), "interpreter": sys.executable}
                data["env_config"] = environment
                write_json(run_dir / "environment.json", environment)
                return environment

            environment = phase("BUILD_ENV", "EnvBuilder", plan_environment, "prepare_environment_plan")
            if input_data.get("prepare_only"):
                data["execution"] = {"mode": "repository", "executed": False,
                                     "not_runnable": True, "reason": "仅准备：未安装依赖、未执行训练"}
                data["validation"] = {"status": "prepared", "is_reproduced": None,
                                      "result_level": "prepared", "reason": "版本、真实数据与命令已准备，等待用户执行测试。"}
            else:
                runner = self.runner or RepositoryRunner(logger=self.logger)
                def execution_event(event):
                    if event.get("type") == "repository_environment":
                        emit({"type": "state", "state": "BUILD_ENV", "agent": "EnvBuilder",
                              "phase_id": "prepare_dependencies", "status": event["status"],
                              "reason": event.get("reason") or
                              ("依赖验证通过，环境已就绪" if event["status"] == "success" else
                               "正在检查依赖并自动准备所需环境")})
                    elif event.get("type") == "repository_step" and event.get("status") == "running":
                        emit({"type": "state", "state": "EXECUTE_CODE", "agent": "CodeExecutor",
                              "phase_id": "execute_repository", "status": "running"})
                    emit({**event, "phase_id": "execute_repository"})
                execution = phase("EXECUTE_CODE", "CodeExecutor", lambda: runner.run(
                    snapshot["path"], profile["steps"], environment,
                    use_docker=self.use_docker,
                    on_event=execution_event),
                    "execute_repository")
                data["execution"] = execution
                # Images shipped in the fixed source tree are published figures,
                # not evidence generated by this experiment.
                execution["artifacts"] = [artifact for artifact in execution.get("artifacts", [])
                                          if artifact.get("name") not in snapshot.get("files", {})]
                execution.setdefault("final", {})["artifacts"] = execution["artifacts"]
                records, metric_error = [], ""
                try:
                    records = adapter.metric_records(execution, data["spec_sha256"])
                except ValueError as exc:
                    metric_error = str(exc)
                from src.execution_completion import completed_execution
                data['validation'] = completed_execution(execution)
                data['validation']['metric_records'] = records
                write_json(run_dir / "metrics.json", {"records": records, "error": metric_error,
                                                     "spec_sha256": data["spec_sha256"]})
            optimization_requested = input_data.get("enable_optimization", False) is True
            data["optimization"] = {
                "optimized": False, "status": "not_implemented" if optimization_requested else "disabled",
                "requested": optimization_requested, "available": False,
                "reason": "优化接口尚未开放；本轮保留作者实验协议。" if optimization_requested else
                          "优化未启用；本轮保留作者实验协议。"}
            emit({"type": "state", "state": state, "agent": "Optimizer", "status": "skipped"})
        except Exception as exc:
            error = f"{state} 阶段失败: {exc}"
            if active_phase_id:
                emit({"type": "state", "state": state, "agent": active_agent,
                      "phase_id": active_phase_id, "status": "error", "reason": str(exc)})
            self.logger.log("RepositoryReproduction", state, "ERROR", error)
            data.setdefault("execution", {"mode": "repository", "success": False,
                                           "executed": False, "not_runnable": True, "reason": error})
            if data.get("validation") and data.get("execution", {}).get("executed"):
                data["analysis_status"] = "failed"
                data["analysis_error"] = error
            else:
                data["validation"] = {"status": "execution_failed",
                                      "is_reproduced": False, "result_level": "failed", "reason": error}
        sync_calls()
        data["audit_stats"] = self.logger.get_stats()
        data["report"] = phase("GENERATE_REPORT", "ReportGenerator", lambda: ReportGeneratorAgent(
            self.logger).run(data, report_path=data["report_path"])["report"], "generate_report")
        write_json(run_dir / "result.json", {"data": data, "error": error})
        (run_dir / "report.md").write_text(data["report"], encoding="utf-8")
        return {"state": "ERROR" if error else "COMPLETED", "data": data, "error": error}

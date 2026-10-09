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
import subprocess
import sys
import tarfile
import urllib.request
import uuid
from pathlib import Path

from src.repository_profiles import get_profile
from src.repository_adapters import get_adapter
from src.safety.paths import workspace_path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


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
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip()[-1000:])
    return result.stdout if binary else result.stdout.decode("utf-8", "replace").strip()


def _canonical_url(url):
    return url.rstrip("/").removesuffix(".git").lower()


def export_repository(data_root, profile, destination, *, offline=False):
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
            origin = _git(candidate, "remote", "get-url", "origin")
            commit = _git(candidate, "rev-parse", "--verify", f"{revision}^{{commit}}")
            if _canonical_url(origin) == _canonical_url(url) and commit == revision:
                source = candidate
                break
        except (OSError, subprocess.SubprocessError, RuntimeError):
            continue
    if source is None:
        if offline:
            raise RuntimeError("离线模式中没有来源与固定版本均可核验的仓库缓存")
        # Use a unique temporary directory so concurrent preparations do not share
        # a partly cloned repository or move each other's worktree HEAD.
        cache.parent.mkdir(parents=True, exist_ok=True)
        source = cache.parent / f".fetch-{uuid.uuid4().hex}"
        result = subprocess.run(["git", "clone", "--depth", "1", "--single-branch", url, str(source)],
                                capture_output=True, timeout=300)
        if result.returncode:
            raise RuntimeError("仓库下载失败: " + result.stderr.decode("utf-8", "replace")[-1000:])
        _git(source, "fetch", "--depth", "1", "origin", revision, timeout=120)
        _git(source, "checkout", "--detach", revision)
        if _git(source, "rev-parse", "HEAD") != revision:
            raise RuntimeError("固定仓库版本校验失败，停止执行")
        if not cache.exists():
            try:
                source.rename(cache)
                source = cache
            except OSError:
                pass  # Another request may have populated the same immutable key.
    archive = _git(source, "archive", "--format=tar", revision, binary=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        members = tar.getmembers()
        # Validate the entire archive before creating any extracted files.
        for member in members:
            try:
                workspace_path(dest, member.name, "archive", forbid_git=True)
            except ValueError as exc:
                raise ValueError("仓库归档包含越界路径") from exc
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"首版仓库执行不支持symlink/submodule: {member.name}")
        dest.mkdir(parents=True, exist_ok=False)
        for member in members:
            # Recheck immediately before writing; earlier entries may exist now.
            target = workspace_path(dest, member.name, "archive", forbid_git=True)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = tar.extractfile(member).read()
                target.write_bytes(payload)
                files[member.name] = hashlib.sha256(payload).hexdigest()
    for path in get_adapter(profile).required_files(profile):
        if not workspace_path(dest, path, "required file", forbid_git=True).is_file():
            raise RuntimeError(f"固定仓库缺少必需入口: {path}")
    if (dest / ".gitmodules").exists():
        raise RuntimeError("首版尚不支持submodule仓库")
    return {"url": url, "revision": revision, "resolved_sha": revision,
            "path": str(dest.resolve()), "cache_path": str(source.resolve()),
            "archive_sha256": hashlib.sha256(archive).hexdigest(), "files": files}


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
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "AutoReproducer"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(spec["bytes"] + 1)
        if len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise RuntimeError("真实ETTh1下载内容校验失败；不会替换为合成数据")
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.parent / f".{uuid.uuid4().hex}.part"
        temp.write_bytes(content)
        temp.replace(cache)
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
    return bool(execution.get("success") and final.get("exit_code") == 0
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
    deps = (execution.get("environment") or {}).get("dependencies_path", "")
    env["PYTHONPATH"] = str(deps or "")
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

    def run(self, input_data, on_event=None):
        if get_profile(input_data["experiment_profile"]).get("adapter_id") in {"siren", "neural_ode"}:
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
        analysis_mode = input_data.get("analysis_mode", "multi_agent")
        active_phase_id, active_agent = "", ""
        public_packet = None
        run_dir = self.root / "runs" / f"repository_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True)
        data["run_dir"] = str(run_dir.resolve())
        data["report_path"] = str((run_dir / "report.md").resolve())

        def emit(event):
            if on_event:
                on_event(event)

        def sync_calls():
            nonlocal counted_calls
            actual_calls = max(0, (self.llm.get_call_count() if self.llm else 0) - initial_calls)
            self.logger.add_llm_calls(actual_calls - counted_calls)
            counted_calls = actual_calls
            data["total_llm_calls"] = actual_calls

        def analysis_event(name, event):
            nonlocal state, active_phase_id, active_agent
            mapping = {"reader": ("READ_PAPER", "PaperReader", "analyze_reader"),
                       "finder": ("FIND_RESOURCES", "ResourceFinder", "analyze_finder"),
                       "builder": ("BUILD_ENV", "EnvBuilder", "analyze_builder"),
                       "verifier": ("BUILD_ENV", "Verifier", "review_readiness"),
                       "result_validator": ("VALIDATE", "ResultValidator", "review_result_summary")}
            state, agent, phase_id = mapping[name]
            active_phase_id, active_agent = phase_id, agent
            status = event["status"]
            if status == "started":
                self.logger.begin_plan(f"repository_{name}")
            else:
                self.logger.end_plan(f"repository_{name}")
            emit({"type": "state", "state": state, "agent": agent, "phase_id": phase_id,
                  "attempt": event.get("attempt", 1),
                  **{key: event[key] for key in ("calls", "reason") if key in event},
                  "outcome": None if status == "started" else status,
                  "status": "running" if status == "started" else
                  "success" if status == "accepted" else "error"})

        def require_real_llm():
            if not self.llm or self.llm.mock_mode or not self.llm.base_url or not self.llm.model:
                raise ValueError("真实多Agent分析需要有效的LLM配置")

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
            adapter = get_adapter(profile)
            data.update(paper_title=profile["paper"]["title"], paper_info=profile["paper"],
                        experiment_spec=profile, spec_sha256=spec_digest(profile))
            prepare_only = bool(input_data.get("prepare_only"))
            multi_analysis = bool(input_data.get("use_llm_review")) and analysis_mode == "multi_agent" and not prepare_only
            public_protocol_review = bool(input_data.get("use_llm_review")) and analysis_mode == "public_protocol" and not prepare_only
            stages = [
                ("select_experiment", "PaperReader", "选择实验", "读取冻结论文协议和验收范围", True),
                ("prepare_repository", "ResourceFinder", "准备源码与数据", "导出固定作者源码并核验真实数据来源", True),
                ("prepare_environment_plan", "EnvBuilder", "准备环境计划", "核对解释器和冻结依赖、命令；尚未安装依赖", True),
                ("load_public_sources", "SourceLoader", "加载公开证据", "收集论文原文、作者代码和公开说明", multi_analysis),
                ("analyze_reader", "PaperReader", "分析论文协议", "依据公开原文核对实验协议与参考指标", multi_analysis),
                ("analyze_finder", "ResourceFinder", "分析资源映射", "核对固定源码、实验入口与数据映射", multi_analysis),
                ("analyze_builder", "EnvBuilder", "分析依赖方案", "区分作者原始依赖与尚未执行的兼容建议", multi_analysis),
                ("review_readiness", "Verifier", "训练前证据预审", "检查前三项公开分析；通过后才允许真实执行", multi_analysis),
                ("execute_repository", "CodeExecutor", "代码执行", "按本次执行计划运行代码并记录实际结果", not prepare_only),
                ("verify_protocol", "Verifier", "训练后本地核验", "核验完整运行协议并独立复算最终指标", not prepare_only and profile["validation"]["level"] == "reference"),
                ("validate_metrics", "ResultValidator", "确定性数值验收", "按冻结协议判定结果；阶段完成与复现通过分别显示", not prepare_only),
                ("review_result_summary", "ResultValidator", "解释结果摘要", "仅在授权后用 API 解释数值摘要，不改变本地判定", multi_analysis and bool(input_data.get("allow_result_summary_review"))),
            ]
            if analysis_mode == "public_protocol":
                stages.append(("review_public_protocol", "Verifier", "解释公开协议", "仅用 API 解释公开论文与作者协议", public_protocol_review))
            stages.append(("generate_report", "ReportGenerator", "生成报告", "汇总执行证据、本地判定与本项实验的范围", True))
            emit({"type": "pipeline_plan", "pipeline": "repository", "stages": [
                {"id": identifier, "agent": agent, "title": title, "description": description,
                 "status": "waiting" if enabled else "skipped"}
                for identifier, agent, title, description, enabled in stages]})
            write_json(run_dir / "experiment_spec.json", {"spec": profile, "sha256": data["spec_sha256"]})
            offline = bool(input_data.get("offline"))

            def prepare():
                snapshot = export_repository(self.root, profile, run_dir / "repo", offline=offline)
                data["repository"] = snapshot
                write_json(run_dir / "repository.json", snapshot)
                dataset = adapter.prepare_dataset(self.root, profile, run_dir / "repo", offline=offline)
                data["dataset_provenance"] = dataset
                write_json(run_dir / "dataset.json", dataset)
                data["resources"] = {"code_repo_url": snapshot["url"], "dataset_url": dataset["url"],
                                     "confidence": 1.0}
                return snapshot

            snapshot = phase("FIND_RESOURCES", "ResourceFinder", prepare, "prepare_repository")

            def plan_environment():
                if sys.version_info[:2] not in {(3, 11), (3, 12)}:
                    raise RuntimeError("DLinear兼容预设目前支持Python 3.11/3.12，请切换解释器后重试")
                plan = {"profile": profile["id"], "workspace": snapshot["path"],
                        "spec_sha256": data["spec_sha256"], "steps": profile["steps"]}
                data["execution_plan"] = plan
                write_json(run_dir / "execution_plan.json", plan)
                environment = {**profile["environment"], "python_version": platform.python_version(),
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
                if input_data.get("use_llm_review") and analysis_mode == "multi_agent":
                    from src.repository_analysis import RepositoryAnalysis, RepositoryAnalysisError
                    from src.repository_public_sources import RepositoryPublicSources
                    require_real_llm()
                    sources = adapter.public_sources(self.root)
                    public_packet = phase("READ_PAPER", "SourceLoader", lambda: sources.build_packet(
                        snapshot["path"], snapshot, profile, offline=offline), "load_public_sources")
                    data["public_source_manifest"] = str(sources.manifest_path)
                    write_json(run_dir / "public_sources.json", public_packet)
                    try:
                        data["repository_analysis"] = adapter.analysis(self.llm, self.logger, max_repairs=1).run(
                            public_packet, profile, on_stage=analysis_event)
                    except RepositoryAnalysisError as exc:
                        data["repository_analysis"] = exc.result
                        raise
                    finally:
                        sync_calls()
                        if data.get("repository_analysis"):
                            write_json(run_dir / "repository_analysis.json", data["repository_analysis"])
                runner = self.runner or RepositoryRunner(logger=self.logger)
                execution = phase("EXECUTE_CODE", "CodeExecutor", lambda: runner.run(
                    snapshot["path"], profile["steps"], environment,
                    use_docker=self.use_docker,
                    on_event=lambda execution_event: emit({**execution_event, "phase_id": "execute_repository"})),
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
                if not metric_error and profile["validation"]["level"] == "reference":
                    from src.repository_validation import verify_dlinear_protocol
                    def verify_local_execution():
                        execution["protocol_verification"] = adapter.verify_protocol(
                            profile, execution, snapshot["path"], snapshot, data["dataset_provenance"])
                        execution["independent_metrics"] = adapter.recompute_metrics(
                            execution, snapshot["path"], data["dataset_provenance"], records)
                        passed = (execution["protocol_verification"].get("pass") is True and
                                  execution["independent_metrics"].get("pass") is True)
                        return {"success": passed, "outcome": "pass" if passed else "fail",
                                "reason": "完整运行协议和独立指标复算均通过" if passed else
                                "完整运行协议或独立指标复算未通过"}

                    phase("VALIDATE", "Verifier", verify_local_execution, "verify_protocol")
                    if execution["independent_metrics"].get("pass"):
                        from src.execution_artifacts import collect_images
                        artifact_result = collect_images(execution["independent_metrics"]["artifact_dir"],
                                                         "full", self.logger.session_id)
                        execution.setdefault("artifacts", []).extend(artifact_result["artifacts"])
                        execution.setdefault("artifact_warnings", []).extend(
                            artifact_result.get("artifact_warnings", []))
                        execution["final"]["artifacts"] = execution["artifacts"]
                    write_json(run_dir / "protocol_verification.json", execution["protocol_verification"])
                    write_json(run_dir / "independent_metrics.json", execution["independent_metrics"])
                elif profile["validation"]["level"] == "reference":
                    emit({"type": "state", "state": "VALIDATE", "agent": "Verifier",
                          "phase_id": "verify_protocol", "status": "blocked",
                          "reason": "无法解析有效最终指标，本地核验未执行"})
                data["validation"] = phase("VALIDATE", "ResultValidator", lambda: adapter.validate(
                    profile, execution, records, metric_error), "validate_metrics")
                write_json(run_dir / "metrics.json", {"records": records, "error": metric_error,
                                                     "spec_sha256": data["spec_sha256"]})
                if input_data.get("use_llm_review") and analysis_mode == "multi_agent":
                    data["analysis_status"] = "public_readiness_accepted"
                    if input_data.get("allow_result_summary_review"):
                        summary = {"metrics": {record["name"]: record["value"] for record in records},
                                   "epochs_completed": (execution.get("protocol_verification") or {}).get("epochs_completed"),
                                   "protocol_pass": (execution.get("protocol_verification") or {}).get("pass") is True,
                                   "independent_metrics_pass": (execution.get("independent_metrics") or {}).get("pass") is True}
                        from src.repository_analysis import RepositoryAnalysis, RepositoryAnalysisError
                        try:
                            data["result_analysis"] = phase("VALIDATE", "ResultValidator", lambda:
                                adapter.analysis(self.llm, self.logger).review_result_summary(
                                    summary, public_packet, profile, on_stage=analysis_event), "review_result_summary")
                            data["analysis_status"] = "completed"
                        except RepositoryAnalysisError as exc:
                            data["result_analysis"] = exc.result
                            data["analysis_status"] = "result_analysis_failed"
                            data["analysis_error"] = str(exc)
                        finally:
                            sync_calls()
                            write_json(run_dir / "result_analysis.json", data["result_analysis"])
                if input_data.get("use_llm_review") and analysis_mode == "public_protocol":
                    state = "VALIDATE"
                    active_phase_id, active_agent = "review_public_protocol", "Verifier"
                    emit({"type": "state", "state": state, "agent": active_agent,
                          "phase_id": active_phase_id, "status": "running"})
                    if not self.llm or self.llm.mock_mode or not self.llm.base_url or not self.llm.model:
                        raise ValueError("真实API分析需要有效的LLM配置")
                    # Only published facts go to the API. Local logs, workspace
                    # paths, dataset contents, checkpoints and measured results
                    # remain local and never enter this prompt.
                    public_sources = {
                        "paper": "https://arxiv.org/html/2205.13504v3#S5.T2",
                        "published_table_row": "Table 2, multivariate ETTh1, horizon 96, DLinear: MSE 0.375; MAE 0.399.",
                        "implementation_details": "https://arxiv.org/html/2205.13504v3#A2.SS2; LTSF-Linear input length 336, moving average window 25.",
                        "author_seed_statement": "https://github.com/cure-lab/LTSF-Linear/issues/33#issuecomment-1331937601; 作者：我们论文里只跑了一个seed。设备和torch可能导致差异。",
                        "author_script": "https://github.com/cure-lab/LTSF-Linear/blob/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6/scripts/EXP-LongForecasting/Linear/etth1.sh; seq_len=336, pred_len=96, batch_size=32, learning_rate=0.005, itr=1.",
                    }
                    prompt = ("从以下公开论文及作者实现摘录中解析DLinear ETTh1 336→96单项实验的验收协议。"
                              "只使用这些公开事实，不猜测实际训练结果，不宣称整篇论文全部实验已经复现。"
                              "返回JSON：{\"summary\":\"中文协议说明\",\"reference_metrics\":{\"mse\":数值,\"mae\":数值},"
                              "\"limitations\":[\"该单项实验的实际范围限制\"]}。\n" +
                              json.dumps(public_sources, ensure_ascii=False))
                    raw_review = self.llm.chat(prompt, task="repository_public_protocol", temperature=0.1)
                    if raw_review.startswith("[LLM API Error"):
                        raise RuntimeError("真实API分析调用失败")
                    try:
                        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_review.strip())
                        review = json.loads(cleaned)
                        if not isinstance(review.get("summary"), str) or not review["summary"].strip():
                            raise ValueError("API返回缺少分析")
                        if review.get("reference_metrics") != {"mse": 0.375, "mae": 0.399}:
                            raise ValueError("API解析的公开参考值不符合已核对论文")
                    except (ValueError, TypeError, AttributeError) as exc:
                        raise ValueError("真实API分析返回结构无效") from exc
                    data["llm_analysis"] = {**review, "model": self.llm.model,
                                            "usage": self.llm.last_usage, "source": "real_api",
                                            "input_scope": "public_paper_only"}
                    sync_calls()
                    self.logger.log("Verifier", "repository_api_review", "SUCCESS", "真实API分析已完成")
                    emit({"type": "state", "state": "VALIDATE", "agent": "Verifier",
                          "phase_id": "review_public_protocol", "status": "success", "outcome": "accepted"})
                    write_json(run_dir / "llm_analysis.json", data["llm_analysis"])
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
                data["validation"] = {"status": "analysis_failed" if data.get("repository_analysis") else "execution_failed",
                                      "is_reproduced": False, "result_level": "failed", "reason": error}
        sync_calls()
        data["audit_stats"] = self.logger.get_stats()
        data["report"] = phase("GENERATE_REPORT", "ReportGenerator", lambda: ReportGeneratorAgent(
            self.logger).run(data, report_path=data["report_path"])["report"], "generate_report")
        write_json(run_dir / "result.json", {"data": data, "error": error})
        (run_dir / "report.md").write_text(data["report"], encoding="utf-8")
        return {"state": "ERROR" if error else "COMPLETED", "data": data, "error": error}

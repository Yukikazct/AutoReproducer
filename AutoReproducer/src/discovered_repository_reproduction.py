"""Run a source-grounded experiment from an uploaded paper's author repository.

No paper identity is registered here. The frozen plan is derived from the actual
PDF and pinned repository, then executed with the repository process owner.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from filelock import FileLock

from src.dependency_cache import AUTO_PREPARATION_LOCK_TIMEOUT
from src.execution_plan import build_plan
from src.method_adapters import digest, read_json, write_json
from src.pdf_input import extract_pdf_input
from src.preset_downloads import atomic_cache_bytes, download_bytes
from src.process_lifecycle import termination_signals
from src.repository_evidence import normalize_repository_url
from src.repository_reproduction import export_repository, spec_digest
from src.repository_runner import RepositoryRunner
from src.runtime_preparation import run_owned_process
from src.safety.paths import workspace_path


_MAX_DATA_BYTES = 512 * 1024 * 1024
_REF = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,200}")
_SHA = re.compile(r"[0-9a-f]{40}")


def author_repository_input(request):
    """Re-read the actual PDF; preview flags alone never authorize execution."""
    if request.get("mock_mode") or request.get("code") or request.get("corpus_paper"):
        raise ValueError("作者仓库复现需要真实 PDF 输入，不能混用模拟、外部代码或语料预设")
    if not request.get("pdf_path"):
        raise ValueError("缺少用于核验作者代码声明的原始 PDF")
    document = extract_pdf_input(request["pdf_path"])
    expected = (request.get("pdf_input") or {}).get("sha256")
    if expected and expected != document.sha256:
        raise ValueError("PDF 在资源发现后发生变化，不能继续原仓库计划")
    resources = request.get("resources") or {}
    selected = normalize_repository_url((resources.get("repo_discovery") or {}).get("selected_repo")
                                        or resources.get("code_repo_url") or resources.get("selected_repo") or "")
    if not selected:
        raise ValueError("没有有效的作者 GitHub 仓库")
    evidence = next((link for link in document.repository_links
                     if link.get("is_author_code") is True
                     and link.get("evidence_type") == "author_code_statement"
                     and link.get("source") in {"pdf_text", "pdf_annotation"}
                     and link["url"].casefold() == selected.casefold()), None)
    if evidence is None:
        raise ValueError("原始 PDF 的作者代码声明没有确认所选仓库")
    return document, selected, deepcopy(evidence)


def resolve_repository_revision(url, requested="", *, offline=False):
    """Pin the current public ref exactly; never silently retain another HEAD."""
    requested = str(requested or "")
    if _SHA.fullmatch(requested):
        return requested, {"requested": requested, "resolved_sha": requested, "source": "explicit_commit"}
    if offline:
        raise ValueError("离线作者仓库执行需要预先固定完整 commit SHA")
    ref = requested or "HEAD"
    if not _REF.fullmatch(ref) or ".." in ref or ref.endswith("/"):
        raise ValueError("无效的仓库版本引用")
    patterns = [ref] if ref == "HEAD" or ref.startswith("refs/") else [
        f"refs/heads/{ref}", f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"]
    completed = run_owned_process(["git", "ls-remote", url, *patterns], timeout_s=60)
    if completed.returncode:
        raise RuntimeError("无法从所选作者仓库解析固定版本；未使用其他 HEAD")
    refs = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and _SHA.fullmatch(parts[0]) and parts[1] in patterns:
            refs[parts[1]] = parts[0]
    if f"refs/tags/{ref}^{{}}" in refs:
        refs.pop(f"refs/tags/{ref}", None)
    commits = set(refs.values())
    if len(commits) != 1:
        raise ValueError("作者仓库引用不存在或对应多个不同版本，不能固定本次实验")
    revision = commits.pop()
    return revision, {"requested": requested, "resolved_ref": ref, "resolved_sha": revision,
                      "source": "git_ls_remote"}


def prepare_plan_datasets(root, workspace, plan, snapshot, *, offline=False):
    """Stage only quoted real inputs; bundled data must belong to the Git tree."""
    records = []
    for index, dataset in enumerate(plan.get("datasets", []), 1):
        kind = dataset.get("kind")
        if kind == "bundled":
            files = {}
            for name in dataset.get("paths", []):
                expected = snapshot["files"].get(name)
                path = workspace_path(workspace, name, "bundled dataset", must_exist=True, forbid_git=True)
                if not expected or not path.is_file() or not path.stat().st_size or digest(path) != expected:
                    raise ValueError(f"仓库内数据未通过固定源码校验: {name}")
                files[name] = expected
            if not files:
                raise ValueError("仓库数据声明没有实际文件")
            records.append({"kind": kind, "files_sha256": files, "verified": True,
                            "source": snapshot["url"], "revision": snapshot["resolved_sha"]})
        elif kind == "https":
            from urllib.parse import urlsplit
            url, expected = dataset.get("url", ""), dataset.get("sha256", "")
            size = dataset.get("bytes")
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                    or not re.fullmatch(r"[a-f0-9]{64}", expected)
                    or type(size) is not int or not 0 < size <= _MAX_DATA_BYTES):
                raise ValueError("真实数据下载缺少 HTTPS 来源、固定 SHA-256 或合理大小")
            target = workspace_path(workspace, dataset["target"], "downloaded dataset", forbid_git=True)
            if dataset["target"] in snapshot["files"] or target.exists():
                raise ValueError("数据下载不能覆盖作者仓库已有文件")
            cache = workspace_path(root, f"dataset_cache/discovered/{expected}/input", "dataset cache")
            cached = cache.is_file() and cache.stat().st_size == size and digest(cache) == expected
            if not cached:
                if offline:
                    raise RuntimeError("离线缓存缺少本次固定真实数据")
                content = download_bytes(url, max_bytes=size, timeout_s=60)
                if len(content) != size or hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("真实数据下载的完整大小或 SHA-256 不匹配")
                atomic_cache_bytes(cache, content)
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_cache_bytes(target, cache.read_bytes())
            records.append({"kind": kind, "url": url, "canonical_url": url,
                            "download_source": None if cached else url, "cache_hit": cached,
                            "target": dataset["target"], "sha256": expected, "bytes": size,
                            "verified": True})
        else:
            raise ValueError("本次计划没有受支持的真实数据来源；不会生成合成替代数据")
    if not records:
        raise ValueError("没有完整、可校验的实验数据声明")
    return {"kind": "real", "sources": records, "verified": True,
            "sha256": spec_digest(records), "split": "由原文引用冻结的作者实验划分"}


def verify_snapshot(workspace, snapshot, generated):
    for name, expected in {**snapshot["files"], **generated}.items():
        path = workspace_path(workspace, name, "frozen execution source", must_exist=True, forbid_git=True)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"执行后固定源码或适配文件改变: {name}")


def environment_for_plan(plan):
    from src.discovered_repository_plan import CPU_COMPATIBILITY_PINS, CPU_INDEX_URL
    requirements = plan.get("requirements") or {}
    pins = requirements.get("compatibility") or []
    checked = []
    for value in pins:
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+-]*)", value)
        if not match:
            raise ValueError("兼容环境必须使用已审阅的精确 CPU 依赖版本")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        known = {re.sub(r"[-_.]+", "-", key).lower(): version for key, version in CPU_COMPATIBILITY_PINS.items()}
        expected = str(known.get(name, ""))
        # The public map may store versions or complete requirements.
        expected = expected.split("==")[-1]
        if match[2] != expected:
            raise ValueError(f"计划依赖不在已审阅兼容环境中: {name}")
        checked.append(value)
    if not checked:
        raise ValueError("缺少冻结的本次执行依赖")
    return {"requirements_txt": f"--extra-index-url {CPU_INDEX_URL}\n" + "\n".join(checked) + "\n",
            "author_requirements": deepcopy(requirements.get("author", [])),
            "compatibility_reason": requirements.get("reason", ""),
            "note": "基于作者依赖与源码导入选择的项目 CPU 兼容环境；不是论文原始运行环境的等价声明。",
            "auto_prepare": True, "dependency_health_check": True,
            "cache_lock_timeout_s": AUTO_PREPARATION_LOCK_TIMEOUT}


_PREFLIGHT = '''import importlib, importlib.metadata, json, platform, sys
from pathlib import Path
root = Path(__file__).resolve().parent
requirements = json.loads((root / "requirements.json").read_text(encoding="utf-8"))
aliases = {"scikit-learn": "sklearn", "pillow": "PIL"}
modules, versions = {}, {}
for requirement in requirements:
    name, expected = requirement.split("==", 1)
    actual = importlib.metadata.version(name)
    if actual != expected:
        raise ValueError("Frozen dependency version mismatch: " + name)
    module = importlib.import_module(aliases.get(name.lower(), name.replace("-", "_")))
    modules[name] = str(Path(module.__file__).resolve())
    versions[name] = actual
import torch, numpy
if torch.cuda.is_available():
    raise ValueError("Generic CPU contract unexpectedly exposes a CUDA device")
x = torch.tensor([[1., 2.]], requires_grad=True)
(x @ x.T).sum().backward()
if not numpy.isfinite(x.grad.numpy()).all():
    raise ValueError("Native CPU operation failed")
record = {"versions": versions, "modules": modules, "python": sys.version,
          "platform": platform.platform(), "device": "cpu", "native_probe_pass": True}
(root / "preflight.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record), flush=True)
'''


class DiscoveredRepositoryReproduction:
    def __init__(self, root, logger, use_docker=False, llm=None):
        self.root, self.logger, self.use_docker, self.llm = Path(root), logger, use_docker, llm
        self.runner = RepositoryRunner(logger=logger)

    @termination_signals()
    def run(self, request, on_event=None):
        from src.run_lifecycle import cancellation_signals
        run_dir = self.root / "runs" / f"repository_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True)
        status = {"status": "running", "pid": os.getpid(), "started_at": time.time()}
        with FileLock(str(run_dir / ".run.lock")), cancellation_signals():
            write_json(run_dir / "run_status.json", status)
            try:
                result = self._run(request, on_event, run_dir)
                status["status"] = "failed" if result.get("error") else "completed"
                return result
            except (KeyboardInterrupt, SystemExit):
                status.update(status="interrupted", reason="作者仓库实验已中断")
                raise
            except Exception:
                status.update(status="failed", reason="作者仓库实验未返回完整结果")
                raise
            finally:
                status["finished_at"] = time.time()
                write_json(run_dir / "run_status.json", status)

    def _run(self, request, on_event, run_dir):
        from src.agents.report_generator import ReportGeneratorAgent
        from src.discovered_repository_plan import build_evidence_packet, propose_plan
        started = time.monotonic()
        emit = on_event or (lambda event: None)
        phases = [("prepare_repository", "FIND_RESOURCES", "ResourceFinder", "核验 PDF 作者来源与固定仓库"),
                  ("plan_repository", "READ_PAPER", "PaperReader", "依据论文与源码冻结实验计划"),
                  ("prepare_dataset", "FIND_RESOURCES", "ResourceFinder", "准备完整真实数据"),
                  ("prepare_environment", "BUILD_ENV", "EnvBuilder", "准备依赖并进行原生运算检查"),
                  ("execute_repository", "EXECUTE_CODE", "CodeExecutor", "运行固定作者入口与独立评估"),
                  ("verify_protocol", "VALIDATE", "Verifier", "核验固定源码、完成证据与独立指标"),
                  ("generate_report", "GENERATE_REPORT", "ReportGenerator", "生成带来源证据的实验报告")]
        emit({"type": "pipeline_plan", "pipeline": "repository", "stages": [
            {"id": identifier, "agent": agent, "title": title, "status": "waiting"}
            for identifier, _, agent, title in phases]})
        def event(identifier, status, reason=""):
            _, state, agent, title = next(item for item in phases if item[0] == identifier)
            self.logger.log(agent, identifier, status.upper(), reason or title)
            if status == "running":
                self.logger.begin_plan(identifier)
            else:
                self.logger.end_plan(identifier)
            emit({"type": "state", "state": state, "agent": agent, "phase_id": identifier,
                  "status": status, "reason": reason or title})
        data = {"run_dir": str(run_dir.resolve()), "report_path": str((run_dir / "report.md").resolve()),
                "paper_info": deepcopy(request.get("paper_info") or {}), "resources": deepcopy(request.get("resources") or {}),
                "verifications": deepcopy(request.get("verifications") or []),
                "fix_records": deepcopy(request.get("fix_records") or []), "repository_executed": False,
                "reproduction_scope": "selected_paper_experiment", "execution_source": "author_repository",
                "optimization": {"mode": "off", "available": False, "optimized": False,
                                 "status": "disabled", "reason": "运行已冻结作者实验协议"}}
        phase, error = "prepare_repository", None
        before_calls = self.llm.get_call_count() if self.llm else 0
        try:
            event(phase, "running")
            if self.use_docker:
                raise ValueError("通用作者仓库执行当前需要本地受控进程模式")
            document, url, evidence = author_repository_input(request)
            data["pdf_input"] = {"sha256": document.sha256, "pages": document.page_count,
                                 "bytes": document.byte_size, "readable": True}
            data["resources"].update(code_repo_url=url, selection_evidence=evidence,
                repository_identity={"status": "paper_linked", "source": "pdf_author_code_statement",
                                     "reason": "原始 PDF 的作者代码声明确认此仓库", "evidence": deepcopy(evidence)})
            pin = (data["resources"].get("repo_discovery") or {}).get("pinned_revision", "")
            offline = bool(request.get("offline"))
            revision, pin_evidence = resolve_repository_revision(url, pin, offline=offline)
            workspace = run_dir / "repo"
            snapshot = export_repository(self.root, {"repository": {"url": url, "revision": revision}},
                                         workspace, offline=offline, required_files=[])
            if snapshot.get("resolved_sha") != revision or snapshot.get("url") != url:
                raise ValueError("导出的仓库与所选作者来源或固定版本不一致")
            data.update(repository=snapshot, revision_resolution=pin_evidence)
            write_json(run_dir / "repository.json", snapshot)
            event(phase, "success")
            phase = "plan_repository"; event(phase, "running")
            packet = build_evidence_packet(request["pdf_path"], workspace, snapshot)
            if packet["pdf"]["sha256"] != document.sha256:
                raise ValueError("原始 PDF 在仓库准备期间发生变化，不能使用不同论文继续规划")
            write_json(run_dir / "source_packet.json", packet)
            planning_error = None
            try:
                plan = propose_plan(self.llm, packet, workspace)
            except BaseException as exc:
                planning_error = exc
                raise
            finally:
                # A rejected proposal may already have requested more source.
                # Keep that exact packet for audit without replacing the
                # original rejection if persisting diagnostic evidence fails.
                try:
                    write_json(run_dir / "source_packet.json", packet)
                except Exception as persistence_error:
                    if planning_error is None:
                        raise
                    data["source_packet_persistence_error"] = str(persistence_error)
            if (plan.get("semantic_review") or {}).get("accepted") is not True:
                raise ValueError("独立的论文与作者协议审查尚未通过，不能开始训练")
            data["grounded_repository_plan"] = plan
            data["spec_sha256"] = spec_digest(plan)
            write_json(run_dir / "experiment_spec.json", {"spec": plan, "sha256": data["spec_sha256"]})
            data["paper_info"]["metrics"] = {item["name"]: item["reference"] for item in plan["metrics"]}
            data["paper_info"]["required_metrics"] = [item["name"] for item in plan["metrics"]]
            environment = environment_for_plan(plan)
            environment["offline"] = offline
            data["env_config"] = deepcopy(environment)
            data["experiment_spec"] = {"id": "discovered_author_experiment", "adapter_id": "discovered",
                "label": "依据本次 PDF 与固定作者源码生成的实验计划", "repository": plan["repository"],
                "paper": deepcopy(data["paper_info"]), "parameters": plan.get("parameters", {}),
                "environment": deepcopy(environment), "validation": {"scope": "selected_paper_experiment",
                    "note": "本次论文与作者源码引文确定实验范围；独立指标与完整执行证据共同决定结果。"}}
            event(phase, "success")
            phase = "prepare_dataset"; event(phase, "running")
            datasets = prepare_plan_datasets(self.root, workspace, plan, snapshot, offline=offline)
            data["dataset_provenance"] = datasets
            write_json(run_dir / "dataset.json", datasets)
            event(phase, "success")
            self._execute_frozen(request, data, plan, snapshot, workspace, run_dir, environment, emit, event)
        except (KeyboardInterrupt, SystemExit) as exc:
            phase = getattr(exc, "reproduction_phase", phase)
            data["interrupted"] = True
            if getattr(exc, "execution", None) is not None:
                data["environment_preparation" if phase == "prepare_environment" else "execution"] = exc.execution
            data["validation"] = {"status": "interrupted", "is_reproduced": None, "result_level": "failed",
                                  "reason": "实验已中断，未完成的训练和指标不能当作成功"}
            event(phase, "interrupted", data["validation"]["reason"])
            write_json(run_dir / "result.json", {"state": "ERROR", "data": data, "error": "interrupted"})
            raise
        except Exception as exc:
            phase = getattr(exc, "reproduction_phase", phase)
            error = str(exc)
            if getattr(exc, "planning_evidence", None) is not None:
                data["plan_rejection"] = deepcopy(exc.planning_evidence)
                write_json(run_dir / "plan_rejection.json", data["plan_rejection"])
            data["validation"] = {"status": "execution_failed" if phase in {"execute_repository", "verify_protocol"}
                                  else "insufficient_evidence", "failure_phase": phase, "is_reproduced": None,
                                  "result_level": "failed", "reason": error}
            data.setdefault("execution", {"mode": "repository", "executed": False, "success": False,
                                          "repository_executed": False})
            event(phase, "error", error)
        data["total_llm_calls"] = (self.llm.get_call_count() if self.llm else 0) - before_calls
        self.logger.add_llm_calls(data["total_llm_calls"])
        data["audit_stats"] = self.logger.get_stats()
        data["run_elapsed_s"] = time.monotonic() - started
        event("generate_report", "running")
        try:
            data["report"] = ReportGeneratorAgent(self.logger).run(data, report_path=data["report_path"])["report"]
            (run_dir / "report.md").write_text(data["report"], encoding="utf-8")
            result = {"state": "ERROR" if error else "COMPLETED", "data": data, "error": error}
            write_json(run_dir / "result.json", result)
            event("generate_report", "success")
            return result
        except (KeyboardInterrupt, SystemExit) as exc:
            data["interrupted"] = True
            data["report_error"] = {"phase": "generate_report", "reason": str(exc) or "报告生成已中断"}
            event("generate_report", "error", "报告生成已中断")
            try:
                write_json(run_dir / "result.json", {"state": "ERROR", "data": data, "error": "interrupted"})
            except Exception:
                # Preserve the cancellation even when diagnostic persistence
                # is unavailable; completed training evidence stays in memory.
                pass
            raise
        except Exception as exc:
            error = "报告生成失败：" + str(exc)
            data["report_error"] = {"phase": "generate_report", "reason": str(exc)}
            event("generate_report", "error", error)
            result = {"state": "ERROR", "data": data, "error": error}
            try:
                write_json(run_dir / "result.json", result)
            except OSError:
                pass
            return result

    def _execute_frozen(self, request, data, plan, snapshot, workspace, run_dir, environment, emit, event):
        """Materialize project capture utilities, then run and independently check."""
        from src.discovered_repository_runtime import CAPTURE_DIR, materialize_capture, verify_capture
        phase = "prepare_environment"
        try:
            event(phase, "running")
            if (plan.get("semantic_review") or {}).get("accepted") is not True:
                raise ValueError("独立协议审查未通过")
            if read_json(run_dir / "experiment_spec.json") != {"spec": plan, "sha256": data["spec_sha256"]}:
                raise ValueError("落盘实验协议与冻结的本次计划不一致")
            packet = read_json(run_dir / "source_packet.json")
            canonical = json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != plan["evidence_sha256"]:
                raise ValueError("落盘来源包与冻结的实验依据不一致")
            materialized = materialize_capture(workspace, plan, data["spec_sha256"], snapshot=snapshot)
            generated = materialized["files"]
            runtime_dir = workspace / CAPTURE_DIR
            (runtime_dir / "preflight.py").write_text(_PREFLIGHT, encoding="utf-8")
            write_json(runtime_dir / "requirements.json", plan["requirements"]["compatibility"])
            for name in ("preflight.py", "requirements.json"):
                generated[f"{CAPTURE_DIR}/{name}"] = digest(runtime_dir / name)
            write_json(run_dir / "adapter_manifest.json", {"files": generated,
                       "note": "原样保留作者源码；项目包装器仅记录训练、导出模型及独立评估"})
            verify_snapshot(workspace, snapshot, generated)
            steps = materialized["steps"]
            execution_plan = build_plan(profile="discovered_author_experiment", spec_sha256=data["spec_sha256"],
                workspace=workspace, repository=plan["repository"], dataset=data["dataset_provenance"], steps=steps)
            data["execution_plan"] = execution_plan
            write_json(run_dir / "execution_plan.json", execution_plan)
            write_json(run_dir / "environment.json", environment)
            def verify_frozen_inputs():
                if digest(request["pdf_path"]) != plan["pdf_sha256"]:
                    raise ValueError("原始 PDF 在执行准备或运行期间发生变化")
                if read_json(run_dir / "experiment_spec.json") != {"spec": plan, "sha256": data["spec_sha256"]}:
                    raise ValueError("落盘实验协议在执行期间发生变化")
                if read_json(run_dir / "execution_plan.json") != execution_plan:
                    raise ValueError("落盘执行计划在执行期间发生变化")
                stored_packet = read_json(run_dir / "source_packet.json")
                encoded = json.dumps(stored_packet, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")
                if hashlib.sha256(encoded).hexdigest() != plan["evidence_sha256"]:
                    raise ValueError("落盘来源包在执行期间发生变化")
                verify_snapshot(workspace, snapshot, generated)
                for dataset in data["dataset_provenance"]["sources"]:
                    if dataset["kind"] == "https":
                        path = workspace_path(workspace, dataset["target"], "frozen real data", must_exist=True)
                        if digest(path) != dataset["sha256"] or path.stat().st_size != dataset["bytes"]:
                            raise ValueError("真实下载数据在执行期间发生变化")
            verify_frozen_inputs()
            if request.get("prepare_only"):
                data["execution"] = {"mode": "repository", "executed": False, "success": False,
                                     "repository_executed": False, "reason": "仅准备计划，尚未执行作者训练"}
                data["validation"] = {"status": "prepared", "is_reproduced": None,
                                      "result_level": "prepared", "reason": "源码、完整真实数据与协议已冻结，尚未执行训练"}
                event(phase, "success", "执行计划已准备，未安装或训练")
                for identifier in ("execute_repository", "verify_protocol"):
                    event(identifier, "skipped", "仅准备模式")
                return

            def progress(item):
                emit({**item, "phase_id": phase})

            preflight = [{"id": "native_preflight", "kind": "check", "required": True,
                          "argv": ["python", "-u", f"{CAPTURE_DIR}/preflight.py"], "cwd": ".",
                          "timeout_s": 180, "env": {"CUDA_VISIBLE_DEVICES": ""},
                          "artifacts": [f"{CAPTURE_DIR}/preflight.json"]}]
            preparation = self.runner.run(workspace, preflight, environment, on_event=progress)
            data["environment_preparation"] = preparation
            write_json(run_dir / "environment_preparation.json", preparation)
            if preparation.get("success") is not True:
                raise RuntimeError("依赖准备或原生导入/运算检查失败：" + str(
                    preparation.get("reason") or preparation.get("final", {}).get("stderr", ""))[-2000:])
            probe = read_json(runtime_dir / "preflight.json")
            if probe.get("native_probe_pass") is not True or probe.get("device") != "cpu":
                raise ValueError("原生 CPU 运算检查缺少成功证据")
            data["native_preflight"] = probe
            verify_frozen_inputs()
            event(phase, "success")
            if request.get("prepare_environment"):
                data["execution"] = {"mode": "repository", "executed": False, "success": False,
                                     "repository_executed": False, "reason": "环境已准备，尚未执行作者训练"}
                data["validation"] = {"status": "environment_ready", "is_reproduced": None,
                                      "result_level": "prepared", "reason": "真实数据、协议和原生环境已准备，尚未执行训练"}
                for identifier in ("execute_repository", "verify_protocol"):
                    event(identifier, "skipped", "仅准备实验环境")
                return
            phase = "execute_repository"; event(phase, "running")
            verify_frozen_inputs()
            # Preparation does not consume the separately frozen author runtime
            # deadline. The fresh evaluator has its own bounded 300-second step.
            prepared_environment = {**environment, "auto_prepare": False, "require_prepared": True,
                "deadline_monotonic": time.monotonic() + sum(step["timeout_s"] for step in steps) + 2}
            execution = self.runner.run(workspace, execution_plan, prepared_environment, on_event=progress)
            data["execution"] = execution
            training_id = next(step["id"] for step in plan["steps"] if step["kind"] == "train")
            data["repository_executed"] = any(step.get("id") == training_id and step.get("executed")
                                               for step in execution.get("steps", []))
            execution["repository_executed"] = data["repository_executed"]
            execution["artifacts"] = [item for item in execution.get("artifacts", [])
                                       if item.get("name") not in snapshot["files"]]
            execution.setdefault("final", {})["artifacts"] = execution["artifacts"]
            write_json(run_dir / "execution.json", execution)
            if execution.get("success") is not True:
                raise RuntimeError("作者训练或独立评估未完整结束：" + str(execution.get("reason") or
                    execution.get("final", {}).get("stderr", ""))[-2000:])
            event(phase, "success")
            phase = "verify_protocol"; event(phase, "running")
            verify_frozen_inputs()
            validation = verify_capture(plan, workspace, execution, data["spec_sha256"])
            data["validation"] = validation
            execution["protocol_verification"] = {"pass": True, **validation["capture"]}
            execution["independent_metrics"] = validation["independent_metrics"]
            write_json(run_dir / "protocol_verification.json", execution["protocol_verification"])
            write_json(run_dir / "independent_metrics.json", execution["independent_metrics"])
            write_json(run_dir / "metrics.json", {"records": validation["comparisons"],
                       "spec_sha256": data["spec_sha256"]})
            write_json(run_dir / "execution.json", execution)
            event(phase, "success", validation["reason"])
        except BaseException as exc:
            exc.reproduction_phase = phase
            raise

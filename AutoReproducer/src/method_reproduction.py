"""Run reviewed method adapters with preparation separate from timed execution."""
import time
import uuid
import os
import json
from copy import deepcopy
from pathlib import Path

from filelock import FileLock
from src.method_adapters import write_json, read_json, digest
from src.method_budget import MAX_OPTIMIZATION_SECONDS, optimization_budget_error
from src.execution_plan import build_plan, plan_step_ids
from src.repository_adapters import get_adapter
from src.repository_profiles import get_profile
from src.repository_runner import RepositoryRunner
from src.dependency_cache import AUTO_PREPARATION_LOCK_TIMEOUT
from src.process_lifecycle import termination_signals
from src.optimization_recovery import recover_interrupted_optimizations, persist_interrupted_run


def method_review_project_sources(workspace, manifest, *, environment_verified=False):
    """Read only the generated, hash-verified adapter files, never an uploaded PDF."""
    root, sources = Path(workspace), []
    for name, identifier in (("run_experiment.py", "project_training_adapter"),
                             ("evaluate.py", "project_independent_evaluator"),
                             ("experiment.json", "project_materialized_parameters")):
        expected = manifest.get("files", {}).get(name)
        if not expected:
            continue
        path = root / name
        if digest(path) != expected:
            raise ValueError(f"在线审核适配证据校验失败: {name}")
        sources.append({"source_id": identifier, "origin": "project_adapter",
                        "url": f"project://{name}", "locator": name, "sha256": expected,
                        "content": path.read_text(encoding="utf-8")})
    imported_path = root / "import_provenance.json"
    if environment_verified and imported_path.is_file():
        imported = read_json(imported_path)
        evidence = {"status": "native_import_and_device_checks_passed", "training_started": False,
                    "frozen_import_paths_verified": True,
                    "module_names": sorted(imported.get("modules", {})),
                    **{key: imported[key] for key in ("torch", "torchvision", "cuda") if key in imported}}
        if "author_package" in imported:
            evidence["author_package_matches_pinned_workspace"] = True
        sources.append({"source_id": "project_environment_preflight", "origin": "project_adapter",
                        "evidence_kind": "measured_environment_preflight",
                        "url": "project://import_provenance.json", "locator": "import_provenance.json (sanitized)",
                        "content": json.dumps(evidence, ensure_ascii=False, indent=2)})
    return sources


class MethodReproduction:
    def __init__(self, root, logger, runner=None, llm=None):
        self.root, self.logger = Path(root), logger
        self.runner, self.llm = runner or RepositoryRunner(logger=logger), llm

    @termination_signals()
    def run(self, request, on_event=None):
        from src.run_lifecycle import cancellation_signals
        # Reject impossible requests and recover abandoned studies before this run
        # claims any record, so a refused request leaves no run directory behind.
        prepare_only, prepare_env = bool(request.get("prepare_only")), bool(request.get("prepare_environment"))
        if prepare_only and prepare_env:
            raise ValueError("仅准备源码与准备完整环境不能同时选择")
        mode = request.get("optimization_mode") or ("validate" if request.get("enable_optimization") else "off")
        if mode not in {"off", "suggest", "validate"}:
            raise ValueError("未知优化模式")
        budget = request.get("budget_seconds", MAX_OPTIMIZATION_SECONDS)
        if mode == "validate" and not (prepare_only or prepare_env):
            budget_error = optimization_budget_error(budget)
            if budget_error:
                raise ValueError(budget_error)
        if request.get("use_docker"):
            raise ValueError("方法预设需要本地执行")
        recover_interrupted_optimizations(self.root / "runs")
        run_dir = self.root / "runs" / f"repository_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True)
        status = {"status": "running", "pid": os.getpid(), "started_at": time.time()}
        with FileLock(str(run_dir / ".run.lock")), cancellation_signals():
            write_json(run_dir / "run_status.json", status)
            try:
                result = self._run(request, on_event, run_dir)
                status.update(status="interrupted" if result["data"].get("interrupted") else
                              "failed" if result.get("error") else "completed")
                return result
            except KeyboardInterrupt:
                status.update(status="interrupted", reason="实验已中断")
                raise
            except Exception:
                status.update(status="failed", reason="运行未能正常生成报告")
                raise
            finally:
                status["finished_at"] = time.time()
                write_json(run_dir / "run_status.json", status)

    def _run(self, request, on_event, run_dir):
        from src.repository_reproduction import export_repository, spec_digest
        from src.agents.report_generator import ReportGeneratorAgent
        from src.method_advice import suggest
        started = time.monotonic()
        emit = on_event or (lambda e: None)
        profile = get_profile(request["experiment_profile"])
        adapter = get_adapter(profile)
        # run() validated this request and created the run directory before _run.
        prepare_only, prepare_env = bool(request.get("prepare_only")), bool(request.get("prepare_environment"))
        mode = request.get("optimization_mode") or ("validate" if request.get("enable_optimization") else "off")
        budget = request.get("budget_seconds", MAX_OPTIMIZATION_SECONDS)
        workspace = run_dir / "repo"
        spec_hash = spec_digest(profile)
        data = {"run_dir": str(run_dir.resolve()), "report_path": str((run_dir / "report.md").resolve()),
                "paper_info": profile["paper"], "paper_title": profile["paper"]["title"],
                "experiment_spec": profile, "spec_sha256": spec_hash, "verifications": [], "fix_records": [],
                "optimization": {"optimized": False, "status": "disabled", "available": True, "mode": mode},
                "analysis_status": "frozen_protocol", "total_llm_calls": 0}
        if request.get("title_resolution"):
            data["title_resolution"] = deepcopy(request["title_resolution"])
        if request.get("pdf_resolution"):
            data["pdf_resolution"] = deepcopy(request["pdf_resolution"])
        before_calls = self.llm.get_call_count() if self.llm else 0
        full_review = bool(request.get("use_llm_review")) and not (prepare_only or prepare_env)
        stages = [("prepare_repository", "FIND_RESOURCES", "ResourceFinder", "核验固定源码与数据"),
                  ("prepare_environment", "BUILD_ENV", "EnvBuilder", "准备或检查实验环境"),
                  ("execute_repository", "EXECUTE_CODE", "CodeExecutor", "代码执行"),
                  ("verify_protocol", "VALIDATE", "Verifier", "核验本次协议与产物"),
                  ("method_advice", "OPTIMIZE", "Optimizer", "智能建议与参数试验"),
                  ("generate_report", "GENERATE_REPORT", "ReportGenerator", "生成实验报告")]
        reviews = [("review_" + role, "READ_PAPER", agent, title) for role, agent, title in (
            ("reader", "PaperReader", "在线分析论文方法"), ("finder", "ResourceFinder", "在线核对资源"),
            ("builder", "EnvBuilder", "在线分析依赖"), ("verifier", "Verifier", "在线证据预审"))]
        stages[2:2] = reviews
        emit({"type": "pipeline_plan", "pipeline": "repository", "stages": [
            {"id": i, "agent": a, "title": title, "description": title,
             "status": "skipped" if ((prepare_only or prepare_env) and i in {"execute_repository", "verify_protocol", "method_advice"})
                       or (i == "method_advice" and mode == "off") or (i.startswith("review_") and not full_review) else "waiting"}
            for i, s, a, title in stages]})
        def event(identifier, status, reason=""):
            _, state, agent, title = next(s for s in stages if s[0] == identifier)
            self.logger.log(agent,identifier,status.upper(),reason or title)
            if status == "running":
                self.logger.begin_plan(identifier)
            else:
                self.logger.end_plan(identifier)
            emit({"type": "state", "state": state, "agent": agent, "phase_id": identifier,
                  "status": status, "reason": reason})
        phase, error = "prepare_repository", None
        execution_started = None
        write_json(run_dir / "experiment_spec.json", {"spec": profile, "sha256": spec_hash})
        try:
            # Reviewed presets fetch and prepare automatically unless the user
            # explicitly requests offline operation. Preparation has its own
            # bounds and does not consume the frozen training/optimization budget.
            offline = bool(request.get("offline"))
            event(phase, "running")
            snapshot = export_repository(self.root, profile, workspace, offline=offline)
            dataset = adapter.prepare_dataset(self.root, profile, workspace, offline=offline)
            manifest = adapter.materialize(workspace, profile, spec_hash)
            sources = adapter.public_sources(workspace, profile)
            data.update(repository=snapshot, dataset_provenance=dataset, adapter_manifest=manifest,
                        resources={"code_repo_url": snapshot["url"], "dataset_url": dataset.get("url", ""), "confidence": 1.0},
                        method_sources=sources, env_config=profile["environment"])
            for name, value in (("repository", snapshot), ("dataset", dataset), ("public_sources", sources)):
                write_json(run_dir / f"{name}.json", value)
            event(phase, "success")
            phase = "prepare_environment"; event(phase, "running")
            steps = adapter.steps(profile, train=not (prepare_only or prepare_env))
            env = {**profile["environment"], "cache_lock_timeout_s": AUTO_PREPARATION_LOCK_TIMEOUT,
                   "dependency_health_check": True, "auto_prepare": True, "offline": offline}
            data["env_config"] = deepcopy(env)
            plan = build_plan(mode="repository", profile=profile["id"], spec_sha256=spec_hash,
                              workspace=snapshot["path"],
                              repository={"url": snapshot.get("url", ""),
                                          "revision": snapshot.get("resolved_sha", "")},
                              dataset={"name": profile["dataset"].get("name", ""),
                                       "sha256": profile["dataset"].get("sha256", "")},
                              limits=profile["budget"], steps=steps)
            data["execution_plan"] = plan
            write_json(run_dir / "execution_plan.json", plan)
            # What lands on disk must describe exactly the steps about to run, so a
            # failed or rewritten plan file cannot change the executed experiment.
            stored = read_json(run_dir / "execution_plan.json")
            if plan_step_ids(stored.get("steps")) != plan_step_ids(plan["steps"]):
                raise RuntimeError("落盘执行计划与本次执行步骤不一致")
            if prepare_only:
                data["execution"] = {"mode": "repository", "executed": False, "not_runnable": True}
                data["validation"] = {"status": "prepared", "result_level": "prepared", "is_reproduced": None,
                                      "reason": "已准备源码、数据与命令；未安装依赖或训练"}
                event(phase, "success")
            else:
                # Check imports, native packages and the requested device before
                # starting any baseline deadline. This retains genuine failure
                # evidence while preventing a failed setup from launching training.
                preparation_steps = steps if prepare_env else adapter.steps(profile, train=False)
                preparation_plan = {**plan, "steps": deepcopy(preparation_steps)}
                data["environment_preparation_plan"] = preparation_plan
                write_json(run_dir / "environment_preparation_plan.json", preparation_plan)
                preparation = self.runner.run(workspace, preparation_steps, env,
                    on_event=lambda execution_event: emit({**execution_event, "phase_id": "prepare_environment"}))
                data["environment_preparation_execution"] = preparation
                data["execution"] = preparation
                data["env_config"] = deepcopy(preparation.get("effective_env_config", env))
                for name in ("dependency_install_attempts", "dependency_health_attempts", "dependency_cache_repairs"):
                    data["env_config"][name] = deepcopy(preparation.get("environment", {}).get(name, []))
                write_json(run_dir / "environment_preparation_execution.json", preparation)
                if not preparation["success"]:
                    raise RuntimeError((preparation.get("final") or {}).get("stderr", "实验环境准备失败")[-1800:])
                adapter.verify_environment(profile, preparation, workspace)
                for name in ("dependency_install_attempts", "dependency_health_attempts", "dependency_cache_repairs"):
                    env[name] = deepcopy(preparation.get("environment", {}).get(name, []))
                data["preparation_elapsed_s"] = time.monotonic() - started
                event(phase, "success")
                if not prepare_env:
                    if full_review:
                        from src.method_advice import review_sources
                        phase = "review_reader"
                        def review_stage(role, status):
                            nonlocal phase
                            phase = "review_" + role
                            event(phase, status)
                        project_sources = method_review_project_sources(workspace, manifest, environment_verified=True)
                        write_json(run_dir / "project_review_sources.json", project_sources)
                        data["method_analysis"] = review_sources(self.llm, profile, sources,
                                                                 review_stage, project_sources=project_sources)
                        write_json(run_dir / "method_analysis.json", data["method_analysis"])
                        data["analysis_status"] = "public_readiness_accepted"
                    execution_started = time.monotonic()
                    if mode == "validate" or full_review:
                        baseline_allowance = min(1200, budget) if mode == "validate" else 1200
                    else:
                        baseline_allowance = profile["budget"]["baseline_s"]
                    baseline_deadline = execution_started + baseline_allowance
                    env.update(require_prepared=True, deadline_monotonic=baseline_deadline,
                               exclude_environment_preparation_from_deadline=True)
                    phase = "execute_repository"; event(phase, "running")
                    execution = self.runner.run(workspace, steps, env,
                        on_event=lambda execution_event, phase_id=phase: emit({**execution_event, "phase_id": phase_id}))
                    preparation_elapsed = execution.get("environment", {}).get("preparation_elapsed_s", 0)
                    execution_started += preparation_elapsed
                    baseline_deadline += preparation_elapsed
                else:
                    execution = preparation
                data["execution"] = execution
                data["env_config"] = deepcopy(execution.get("effective_env_config", env))
                for name in ("dependency_install_attempts", "dependency_health_attempts", "dependency_cache_repairs"):
                    data["env_config"][name] = deepcopy(execution.get("environment", {}).get(name, env.get(name, [])))
                # Only this run's generated artifacts qualify as experiment evidence.
                execution["artifacts"] = [a for a in execution.get("artifacts", []) if a["name"].startswith("artifacts/")]
                execution.setdefault("final", {})["artifacts"] = execution["artifacts"]
                if not execution["success"]:
                    raise RuntimeError((execution.get("final") or {}).get("stderr", "实验执行失败")[-1800:])
                if prepare_env:
                    data["validation"] = {"status": "environment_prepared", "result_level": "prepared",
                                          "is_reproduced": None, "reason": "源码、数据、依赖和设备导入检查完成；尚未训练"}
                else:
                    event(phase, "success")
                    phase = "verify_protocol"; event(phase, "running")
                    data["validation"] = adapter.verify(profile, execution, workspace, snapshot, manifest, spec_hash)
                    data["baseline_elapsed_s"] = time.monotonic() - execution_started
                    if time.monotonic() > baseline_deadline:
                        raise TimeoutError("基线阶段超过冻结时间预算")
                    write_json(run_dir / "validation.json", data["validation"])
                    event(phase, "success")
                    # Publish a complete baseline before calling the external API.
                    data["report"] = ReportGeneratorAgent(self.logger).run(data, report_path=data["report_path"])["report"]
                    (run_dir / "report.md").write_text(data["report"], encoding="utf-8")
                    if mode != "off":
                        phase = "method_advice"; event(phase, "running")
                        remaining = (execution_started + profile["budget"]["total_s"] - time.monotonic() - 15) if mode == "suggest" and not full_review else 45
                        if mode == "validate":
                            from src.method_optimization import validate_candidates
                            data["optimization"] = validate_candidates(self, profile, data, request, execution_started,
                                lambda execution_event, phase_id=phase: emit({**execution_event, "phase_id": phase_id}))
                        else:
                            data["optimization"] = suggest(self.llm, profile, data["validation"], sources, timeout_s=min(45, remaining))
                            data["optimization"]["baseline_run_id"] = run_dir.name
                        event(phase, "success" if data["optimization"]["status"] not in {"advice_timeout", "advice_unavailable"} else "error", data["optimization"].get("reason", ""))
        except (KeyboardInterrupt, SystemExit) as exc:
            # Preserve a verified baseline and the study's partial evidence,
            # while keeping the whole run explicitly incomplete.
            # run_status.json / recover_run classify an interrupted run from this flag.
            data["interrupted"] = True
            optimization_path = run_dir / "optimization.json"
            if optimization_path.is_file():
                data["optimization"] = read_json(optimization_path)
            elif mode != "off" and not (prepare_only or prepare_env):
                data["optimization"].update(status="interrupted", requested=True,
                                            reason="运行已中断，尚未完成建议或优化确认")
                write_json(optimization_path, data["optimization"])
            if getattr(exc, "execution", None) is not None and phase != "method_advice":
                data["execution"] = exc.execution
                if phase == "prepare_environment":
                    data["environment_preparation_execution"] = exc.execution
                    write_json(run_dir / "environment_preparation_execution.json", exc.execution)
            data.setdefault("validation", {"status": "interrupted", "result_level": "failed",
                            "is_reproduced": None, "optimization_eligible": False, "reason": "实验已中断"})
            data["run_elapsed_s"] = time.monotonic() - started
            data["total_llm_calls"] = (self.llm.get_call_count() if self.llm else 0) - before_calls
            persist_interrupted_run(run_dir, "实验已中断，已有记录已保存", data=data)
            event(phase, "error", "实验已中断，已有记录已保存")
            raise
        except Exception as exc:
            error = str(exc)
            analysis = getattr(exc, "analysis", None)
            if isinstance(analysis, dict):
                data["method_analysis"] = analysis
                data["analysis_status"] = "public_readiness_rejected"
                write_json(run_dir / "method_analysis.json", analysis)
            data["validation"] = {"status": "analysis_failed" if phase.startswith("review_") else "execution_failed",
                                  "failure_phase": phase, "result_level": "failed", "is_reproduced": None,
                                  "optimization_eligible": False, "reason": error}
            data.setdefault("execution", {"mode": "repository", "executed": False, "success": False})
            event(phase, "error", error)
        data["total_llm_calls"] = (self.llm.get_call_count() if self.llm else 0) - before_calls
        self.logger.add_llm_calls(data["total_llm_calls"])
        data["audit_stats"] = self.logger.get_stats()
        data["run_elapsed_s"] = time.monotonic() - started
        data["execution_elapsed_s"] = (time.monotonic() - execution_started
                                       if execution_started is not None else 0)
        data["quick_target"] = {"limit_s": profile["budget"]["total_s"], "validated_on_this_run": False,
                                "scope": "prepared_environment_to_report", "elapsed_s": data["execution_elapsed_s"]}
        event("generate_report", "running")
        data["report"] = ReportGeneratorAgent(self.logger).run(data, report_path=data["report_path"])["report"]
        (run_dir / "report.md").write_text(data["report"], encoding="utf-8")
        data["run_elapsed_s"] = time.monotonic() - started
        data["execution_elapsed_s"] = (time.monotonic() - execution_started
                                       if execution_started is not None else 0)
        data["quick_target"]["elapsed_s"] = data["execution_elapsed_s"]
        data["quick_target"]["validated_on_this_run"] = bool(
            not error and profile["adapter_id"] == "siren" and not (prepare_env or prepare_only or full_review) and mode == "suggest"
            and data["execution_elapsed_s"] <= 300 and data["validation"].get("quality_pass")
            and data["optimization"]["status"] == "suggested")
        write_json(run_dir / "result.json", {"data": data, "error": error})
        event("generate_report", "success")
        self.logger.log_experiment("FINISH", "固定方法实验完成", outputs={"title": data["paper_title"]},
                                   result={"state": "ERROR" if error else "COMPLETED", "run_dir": str(run_dir),
                                           "duration_sec": data["run_elapsed_s"], "llm_calls": data["total_llm_calls"]})
        return {"state": "ERROR" if error else "COMPLETED", "data": data, "error": error}

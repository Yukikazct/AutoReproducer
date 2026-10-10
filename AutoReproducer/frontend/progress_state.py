"""Progress state and explicit completion gates (stdlib only)."""
import json
from pathlib import Path
from typing import Any, Dict, List


COMPLETION_RULES = {
    "prepare_runtime": "兼容解释器与应用依赖检查通过，交接结果已返回",
    "prepare_repository": "固定源码、数据及执行计划已准备并完成完整性校验",
    "prepare_environment": "依赖、导入和设备检查返回结果，准备进程已退出并清理",
    "prepare_dependencies": "隔离依赖检查返回结果，安装与检查进程已退出并清理",
    "execute_repository": "本轮全部必需步骤返回结果，执行进程已退出并清理",
    "execute_code": "本轮执行及修复步骤返回结果，执行进程已退出并清理",
    "verify_protocol": "本轮协议、产物与独立指标核验返回结果",
    "method_advice": "建议、候选训练和所需确认返回最终结果，全部所属执行已退出并清理",
    "generate_report": "报告内容已生成，生成结果已返回",
}


def completion_rule(identifier):
    if identifier.startswith(("review_", "analyze_", "verify_")):
        return COMPLETION_RULES.get(identifier, "本阶段分析或核验已返回明确结果")
    return COMPLETION_RULES.get(identifier, "负责人已返回本阶段结果，全部所属执行已退出并清理")


def read_progress_snapshot(path: str) -> Dict[str, Any]:
    """读取并聚合进度文件为前端渲染视图（不可用时返回空视图）。"""
    p = Path(path)
    view: Dict[str, Any] = {
        "state": "INIT", "agent_status": {}, "logs": [],
        "result": None, "done": False, "error": None,
        "running": True, "updated_at": "",
        "execution_output": "", "repository_step": "",
        "execution_context": {}, "repository_step_label": "",
        "active_executions": [],
        "pipeline_stages": [],
        "environment_preparation": {},
        "active_runs": [], "finalization_pending": False,
        "active_workers": [],
        "completion_blockers": [],
        "title_resolution": {},
        "pdf_resolution": {},
    }
    stages: Dict[str, Dict[str, Any]] = {}
    stage_order: List[str] = []
    legacy_latest: Dict[tuple, str] = {}
    has_plan = False
    legacy_repository = False
    execution_numbers: Dict[str, int] = {}
    executions: Dict[tuple, Dict[str, Any]] = {}
    runs: Dict[str, Dict[str, Any]] = {}
    workers: Dict[str, Dict[str, Any]] = {}
    run_end_order = {}
    worker_end_order = {}
    step_end_order = {}
    event_sequence = 0
    terminal_request = None
    output_key = None
    labeled_output = False

    def execution_context(event: Dict[str, Any]) -> Dict[str, Any]:
        identifier = event.get("execution_id")
        identifier = identifier if isinstance(identifier, str) else ""
        index = event.get("step_index")
        index = index if type(index) is int and index > 0 else None
        previous = executions.get((identifier, index if identifier else None), {})
        step_id = event.get("step_id", previous.get("step_id", ""))
        step_id = step_id if isinstance(step_id, str) else ""
        count = event.get("step_count", previous.get("step_count"))
        count = count if type(count) is int and count > 0 and (index is None or count >= index) else None
        number = None
        label = "代码执行"
        if identifier:
            number = execution_numbers.setdefault(identifier, len(execution_numbers) + 1)
            label += f" · 第 {number} 轮"
            if index is not None:
                label += f" · 步骤 {index}" + (f"/{count}" if count is not None else "")
        elif step_id:
            label += " · " + step_id
        return {"execution_id": identifier, "round_number": number, "step_id": step_id,
                "step_index": index, "step_count": count, "label": label,
                "phase_id": event.get("phase_id", previous.get(
                    "phase_id", runs.get(identifier, {}).get("phase_id", ""))),
                "status": event.get("status", previous.get("status", ""))}

    def output_boundary(context: Dict[str, Any]) -> None:
        nonlocal output_key, labeled_output
        identifier = context["execution_id"]
        key = (identifier, context["step_index"] if identifier else context["step_id"])
        # Pure legacy streams keep their original text. A mixed stream gets
        # an explicit anonymous boundary instead of borrowing a newer ID.
        if key != output_key and (identifier or labeled_output):
            view["execution_output"] = (view["execution_output"] + f"\n── {context['label']} ──\n")[-16000:]
        output_key = key
        labeled_output = labeled_output or bool(identifier)

    def apply_stage(identifier: str, event: Dict[str, Any]) -> bool:
        if identifier not in stages:
            stages[identifier] = {"id": identifier, "agent": event.get("agent", ""),
                                  "title": event.get("title", event.get("agent", identifier)),
                                  "description": event.get("description", ""), "status": "waiting"}
            stage_order.append(identifier)
        row = stages[identifier]
        status = event.get("status", "waiting")
        attempt, previous_attempt = event.get("attempt"), row.get("attempt", 1)
        newer_attempt = (type(attempt) is int and type(previous_attempt) is int
                         and attempt > previous_attempt)
        if (type(attempt) is int and type(previous_attempt) is int
                and attempt < previous_attempt):
            return False
        if row.get("_end_sequence") is not None:
            if status == row["status"] and not newer_attempt:
                return False
            if status == "running" and not newer_attempt:
                return False
            if (row["status"] in {"error", "interrupted"} and status == "success"
                    and not newer_attempt):
                return False
        if status == "running":
            # A bounded correction starts a fresh attempt on the same row.
            row.pop("reason", None)
            row.pop("outcome", None)
        row["status"] = status
        row["_end_sequence"] = (event_sequence if status in {
            "success", "error", "interrupted", "skipped", "blocked"} else None)
        for key in ("agent", "state", "attempt", "calls", "reason", "outcome"):
            if key in event:
                row[key] = event[key]
        return True

    def record_scope(scopes, identifier, event, **fields):
        """A closed scope stays closed; failed outcomes cannot become success."""
        previous = scopes.get(identifier, {})
        status = event["status"]
        if previous.get("status") in {"success", "error", "interrupted"}:
            if (status in {"running", "success"}
                    or previous["status"] in {"error", "interrupted"}):
                return False
        previous_outcome = previous.get("reported_status", previous.get("status"))
        if status == "success" and previous_outcome in {"error", "interrupted"}:
            status = previous_outcome
        confirmed = (event.get("cleanup_confirmed", True) is not False
                     or previous.get("status") in {"success", "error", "interrupted"})
        scopes[identifier] = {**previous, **fields,
                              "status": "running" if status != "running" and not confirmed else status}
        if status != "running" and not confirmed:
            scopes[identifier]["reported_status"] = status
        elif status != "running":
            scopes[identifier].pop("reported_status", None)
        return True

    def finish_pending(reason: str, failed: bool) -> None:
        known_failure = any(row["status"] == "error" for row in stages.values())
        for row in stages.values():
            if row["status"] == "waiting":
                row.update(status="blocked", reason=reason)
            elif row["status"] == "running":
                # A pipeline-level failure cannot identify a second failed
                # phase when an explicit phase error is already available.
                # Running also means it started, so never label it unexecuted.
                row.update(status="error" if failed and not known_failure else "blocked",
                           reason="流水线已终止，本阶段未正常结束，未收到完成结果")
                agent = row.get("agent", "")
                if agent and view["agent_status"].get(agent) == "running":
                    view["agent_status"][agent] = row["status"]

    def blockers():
        # An invocation stays open between steps and while its owner cleans up.
        # Empty active-step lists cannot confirm completion of that invocation.
        return ([f"execution:{identifier}" for identifier, run in runs.items()
                 if run["status"] == "running"] +
                [f"worker:{identifier}" for identifier, worker in workers.items()
                 if worker["status"] == "running"] +
                [f"step:{identifier}:{index}" for (identifier, index), context in executions.items()
                 if context["status"] == "running" and runs.get(identifier, {}).get("status") == "running"])

    def finalize(event):
        nonlocal legacy_repository
        if event["type"] == "done":
            view["done"] = True
            view["result"] = event.get("result")
            failed_workers = [worker for worker in workers.values()
                              if worker["status"] in {"error", "interrupted"}]
            if failed_workers and (view["result"] or {}).get("state") != "ERROR":
                message = "后台工作进程未正常结束，不能确认流水线成功"
                view["result"] = {**(view["result"] or {}), "state": "ERROR", "error": message}
            legacy_repository = legacy_repository or bool(
                ((view["result"] or {}).get("data") or {}).get("experiment_spec"))
            view["state"] = (view["result"] or {}).get("state", view["state"]) or view["state"]
            failed = view["state"] == "ERROR"
            if failed:
                view["error"] = (view["result"] or {}).get("error", view["error"])
            finish_pending("上游阶段失败，本阶段未执行" if failed else
                           "流水线已结束，本阶段未执行", failed)
        else:
            view["error"] = event.get("error")
            view["state"] = "ERROR"
            if view["result"] is not None:
                view["result"] = {**view["result"], "state": "ERROR", "error": view["error"]}
            finish_pending("流水线异常终止，本阶段未执行", True)
        view["running"] = False
        view["finalization_pending"] = False

    if not p.exists():
        return view
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return view
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        event_sequence += 1
        if etype == "pipeline_plan":
            planned = ev.get("stages")
            if not isinstance(planned, list):
                continue
            has_plan = True
            planned_order = []
            for stage in planned:
                if not isinstance(stage, dict) or not isinstance(stage.get("id"), str):
                    continue
                identifier = stage["id"]
                if identifier in planned_order:
                    continue
                # select_experiment may already have finished before this
                # profile-specific plan is available; keep observed state.
                observed = stages.get(identifier, {})
                stages[identifier] = {**stage, **observed}
                stages[identifier].setdefault("status", "waiting")
                if not observed and stages[identifier]["status"] in {"success", "error", "interrupted", "blocked"}:
                    # A plan describes intended work; only its owner can report
                    # that work has returned. Explicit skips need no execution.
                    stages[identifier]["status"] = "waiting"
                for key in ("agent", "title", "description"):
                    if key in stage:
                        stages[identifier][key] = stage[key]
                planned_order.append(identifier)
            if "prepare_runtime" in stage_order and "prepare_runtime" not in planned_order:
                planned_order.insert(0, "prepare_runtime")
            stage_order = planned_order + [identifier for identifier in stage_order
                                           if identifier not in planned_order]
        elif etype == "state":
            if not view["running"]:
                continue
            agent = ev.get("agent", "")
            identifier = ev.get("phase_id")
            if identifier:
                if not apply_stage(identifier, ev):
                    continue
            elif agent and not has_plan and ev.get("status") != "waiting":
                # Historical files have no phase IDs. Preserve event
                # order and each running-after-terminal occurrence,
                # rather than pretending that role cards are stages.
                pair = (ev.get("state", ""), agent)
                identifier = legacy_latest.get(pair)
                previous = stages.get(identifier, {}) if identifier else {}
                if not identifier or (ev.get("status") == "running" and
                                      previous.get("status") in {"success", "error", "skipped", "blocked"}):
                    identifier = f"legacy_{len(stages) + 1}"
                    legacy_latest[pair] = identifier
                    title = {"READ_PAPER": "论文解析", "FIND_RESOURCES": "资源准备",
                             "BUILD_ENV": "环境分析与准备", "EXECUTE_CODE": "代码执行",
                             "VALIDATE": "结果验证", "GENERATE_REPORT": "报告生成"}.get(pair[0], pair[0] or agent)
                    if agent.endswith("Optimizer"):
                        title = "智能优化（未启用）" if ev.get("status") == "skipped" else "智能优化"
                    elif agent.endswith("Verifier"):
                        title += "质量核验"
                    elif agent.endswith("SourceLoader"):
                        title = "加载公开证据"
                        legacy_repository = True
                    apply_stage(identifier, {**ev, "title": title,
                        "description": "依据历史进度事件的真实出现顺序恢复"})
                else:
                    apply_stage(identifier, ev)
            view["state"] = ev.get("state", view["state"])
            if agent:
                view["agent_status"][agent] = ev.get("status", "waiting")
        elif etype == "log":
            lg = ev.get("log")
            if isinstance(lg, dict):
                view["logs"].append(lg)
        elif etype == "repository_environment":
            view["environment_preparation"] = ev.get("environment", {})
        elif etype == "title_resolution":
            view["title_resolution"] = {key: ev[key] for key in ("requested_title", "profile", "source", "scope") if key in ev}
        elif etype == "pdf_resolution":
            view["pdf_resolution"] = {key: ev[key] for key in (
                "sha256", "title", "source", "profile", "scope", "pages", "bytes") if key in ev}
        elif etype == "pipeline_worker":
            if not view["running"]:
                continue
            identifier, status = ev.get("worker_id"), ev.get("status")
            if (not isinstance(identifier, str) or not identifier
                    or status not in {"running", "success", "error", "interrupted"}):
                continue
            previous = workers.get(identifier, {})
            if record_scope(workers, identifier, ev, worker_id=identifier,
                            phase_id=ev.get("phase_id", previous.get("phase_id", ""))):
                if workers[identifier]["status"] != "running":
                    worker_end_order[identifier] = event_sequence
        elif etype == "execution_run":
            if not view["running"]:
                continue
            identifier = ev.get("execution_id")
            if not isinstance(identifier, str) or not identifier:
                continue
            status = ev.get("status")
            if status not in {"running", "success", "error", "interrupted"}:
                continue
            previous = runs.get(identifier, {})
            context = execution_context(ev)
            if not record_scope(runs, identifier, ev, execution_id=identifier,
                                phase_id=ev.get("phase_id", previous.get("phase_id", "")),
                                label=context["label"]):
                continue
            status = runs[identifier]["status"]
            if status != "running":
                run_end_order[identifier] = event_sequence
                for step_context in executions.values():
                    if step_context["execution_id"] == identifier and step_context["status"] == "running":
                        # A scope end confirms cleanup, but cannot invent a lost
                        # per-step result. Keep that absence visible in history.
                        step_context["status"] = "unconfirmed" if status == "success" else "interrupted"
        elif etype in {"repository_step", "execution_step"}:
            if not view["running"]:
                continue  # A delayed worker cannot reopen a terminal pipeline.
            context = execution_context(ev)
            key = (context["execution_id"], context["step_index"] if context["execution_id"] else None)
            previous = executions.get(key, {})
            if (context["execution_id"] and context["status"] != "running"
                    and context["status"] == previous.get("status")):
                continue
            if context["execution_id"] and context["status"] == "running" and (
                    runs.get(context["execution_id"], {}).get("status") in {"success", "error", "interrupted"}
                    or previous.get("status") in {"success", "error", "interrupted", "unconfirmed", "skipped"}):
                continue
            if previous.get("status") in {"error", "interrupted"} and context["status"] == "success":
                continue
            view["repository_step"] = ev.get("step_id", "")
            executions[key] = context
            if context["status"] != "running":
                step_end_order[key] = event_sequence
            else:
                step_end_order.pop(key, None)
            if context["status"] == "running":
                output_boundary(context)
            view["execution_context"] = context
            view["repository_step_label"] = context["label"]
            legacy_repository = legacy_repository or etype == "repository_step"
        elif etype in {"repository_output", "execution_output"}:
            # A bounded live viewport; complete streams remain in run logs/report.
            context = execution_context(ev)
            text = ev.get("text", "")
            if text:
                output_boundary(context)
                view["execution_output"] = (view["execution_output"] + text)[-16000:]
        elif etype in {"done", "error"}:
            if (not view["running"] and view["state"] == "ERROR" and etype == "done"
                    and (ev.get("result") or {}).get("state") != "ERROR"):
                continue
            if blockers():
                # A later success request cannot erase an already reported error.
                failed_request = (terminal_request is not None and (
                    terminal_request["type"] == "error"
                    or (terminal_request.get("result") or {}).get("state") == "ERROR"))
                if not failed_request or etype == "error" or (ev.get("result") or {}).get("state") == "ERROR":
                    terminal_request = ev
                view["finalization_pending"] = True
                if etype == "error":
                    view["error"] = ev.get("error")
            else:
                finalize(ev)
        if terminal_request is not None and not blockers():
            finalize(terminal_request)
            terminal_request = None
        view["updated_at"] = ev.get("at", view["updated_at"])
    view["pipeline_stages"] = [stages[identifier] for identifier in stage_order]
    if not view["running"]:
        for context in executions.values():
            if context["status"] == "running":
                context["status"] = "interrupted"
    view["active_executions"] = [context for context in executions.values() if context["status"] == "running"]
    view["active_runs"] = [run for run in runs.values() if run["status"] == "running"]
    view["active_workers"] = [worker for worker in workers.values() if worker["status"] == "running"]
    view["completion_blockers"] = blockers() if view["finalization_pending"] else []
    for row in view["pipeline_stages"]:
        reported = row["status"]
        end_sequence = row.pop("_end_sequence", None)
        row["completion_rule"] = row.get("completion_rule") or completion_rule(row["id"])
        owned_runs = [run for run in view["active_runs"] if run["phase_id"] == row["id"]]
        owned_steps = [context for context in view["active_executions"] if context["phase_id"] == row["id"]]
        owned_workers = [worker for worker in view["active_workers"] if worker["phase_id"] == row["id"]]
        row["completion_pending"] = reported in {"success", "error", "blocked", "interrupted", "skipped"} and bool(owned_runs or owned_steps or owned_workers)
        row["completion_confirmed"] = (reported in {"success", "error", "interrupted", "skipped"}
                                       and (end_sequence is not None or reported == "skipped")
                                       and not row["completion_pending"])
        if row["completion_pending"]:
            row.update(status="running", reported_status=reported,
                       completion_reason="已收到阶段结果，等待所属执行退出并清理")
            if row.get("agent"):
                view["agent_status"][row["agent"]] = row["status"]
        elif reported == "success" and end_sequence is not None:
            failed_after_end = any(run["phase_id"] == row["id"] and run["status"] in {"error", "interrupted"}
                                   and run_end_order.get(identifier, 0) > end_sequence
                                   for identifier, run in runs.items())
            failed_after_end = failed_after_end or any(context["phase_id"] == row["id"] and context["status"] in {"error", "interrupted"}
                                   and step_end_order.get(key, 0) > end_sequence
                                   for key, context in executions.items())
            failed_after_end = failed_after_end or any(worker["phase_id"] == row["id"] and worker["status"] in {"error", "interrupted"}
                                   and worker_end_order.get(identifier, 0) > end_sequence
                                   for identifier, worker in workers.items())
            if failed_after_end:
                row.update(status="error", completion_confirmed=False,
                           completion_reason="所属执行在阶段结果之后失败，不能确认本阶段成功")
                if row.get("agent"):
                    view["agent_status"][row["agent"]] = row["status"]
                if not view["running"] and view["state"] != "ERROR":
                    message = "流水线结束事件与所属执行结果冲突，未确认成功"
                    view.update(state="ERROR", error=message)
                    if view["result"]:
                        view["result"] = {**view["result"], "state": "ERROR", "error": message}
    if not has_plan and legacy_repository:
        for row in view["pipeline_stages"]:
            if row.get("agent", "").endswith("Verifier"):
                if row.get("state") == "BUILD_ENV":
                    row["title"] = "训练前证据预审"
                elif row.get("state") == "VALIDATE":
                    row["title"] = "训练后核验"
    return view

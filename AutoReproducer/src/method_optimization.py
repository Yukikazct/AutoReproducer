"""Small real parameter studies. Selection never consumes holdout results."""
from copy import deepcopy
import math
import time
from pathlib import Path
from filelock import FileLock

from src.method_adapters import write_json, digest, RUNTIMES
from src.method_advice import suggest
from src.repository_adapters import get_adapter
from src.process_lifecycle import termination_signals


class InsufficientBudget(ValueError):
    """A stage cannot start while preserving the frozen reserve; no timeout occurred."""


def check_execution(execution):
    final = execution.get("final") or {}
    if execution.get("cancelled") or final.get("cancelled"):
        interruption = KeyboardInterrupt("实验执行已取消")
        interruption.execution = execution
        raise interruption
    if final.get("timed_out") or final.get("exit_code") == 124:
        raise TimeoutError("实验执行达到时间预算")


def confirmed_gain(method, baseline, candidate):
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (baseline, candidate)):
        return False
    if method == "siren":
        return candidate - baseline >= .1
    return baseline > 0 and (baseline - candidate) / baseline >= .01


class Study:
    def __init__(self, service, profile, parent, started, budget_s, emit):
        self.service, self.profile, self.parent = service, deepcopy(profile), Path(parent)
        self.started, self.deadline, self.emit = started, started + budget_s, emit
        self.adapter = get_adapter(profile)
        self.profile["parameters"]["protocol"] = "pixel_holdout" if profile["adapter_id"] == "siren" else "initial_condition_holdout"
        self.metric = "psnr" if profile["adapter_id"] == "siren" else "mae"
        self.direction = 1 if self.metric == "psnr" else -1
        self.contract = {"base_profile": self.profile, "search_space": profile["search_space"],
                         "selection_split": "validation", "confirmation_split": "holdout", "seeds": [2021,2022],
                         "pixel_split_seed": 1729, "pixel_fractions": [.8,.1,.1],
                         "ode_validation_initials": [[1.5,0.],[0.,1.5]], "ode_holdout_initials": [[1.75,0.],[0.,1.75]],
                         "metric": self.metric, "min_delta": .1 if self.metric == "psnr" else .01,
                         "min_delta_unit": "dB" if self.metric == "psnr" else "relative_reduction",
                         "budget_seconds": budget_s, "reserved_confirmation_s": 2400,
                         "runtime_sha256": digest(RUNTIMES / self.adapter.runtime),
                         "evaluator_sha256": digest(RUNTIMES / self.adapter.evaluator)}
        from src.repository_reproduction import spec_digest
        self.contract_hash = spec_digest(self.contract)
        write_json(self.parent / "optimization_spec.json", {"spec": self.contract, "sha256": self.contract_hash})
        self.runs = {}
        self.selection_closed = False

    def remaining(self):
        return self.deadline - time.monotonic()

    def check_contract(self):
        from src.repository_reproduction import spec_digest
        if (spec_digest(self.contract) != self.contract_hash
                or digest(RUNTIMES / self.adapter.runtime) != self.contract["runtime_sha256"]
                or digest(RUNTIMES / self.adapter.evaluator) != self.contract["evaluator_sha256"]):
            raise ValueError("冻结优化协议或评估器发生变化")

    def train(self, label, *, seed=2021, candidate=None, reserve_s=0):
        from src.repository_reproduction import export_repository, spec_digest
        self.check_contract()
        if self.remaining() <= 0:
            raise TimeoutError("优化总时间预算耗尽")
        if self.remaining() <= reserve_s + 5:
            raise InsufficientBudget("剩余预算不足以启动训练并保留确认预算")
        start = time.monotonic()
        profile = deepcopy(self.profile)
        profile["parameters"]["seed"] = seed
        if candidate:
            parameter, value = candidate["parameter"], candidate["value"]
            if parameter not in profile["search_space"] or value not in profile["search_space"][parameter]:
                raise ValueError("候选超出冻结参数范围")
            profile["parameters"][parameter] = value
        spec_hash = spec_digest(profile)
        directory = self.parent / "trials" / label
        directory.mkdir(parents=True, exist_ok=False)
        workspace = directory / "repo"
        env = {**profile["environment"], "require_prepared": True, "cache_lock_timeout_s": 0,
               "deadline_monotonic": min(self.deadline-reserve_s, start+1200)}
        record = {"label": label, "seed": seed, "candidate": candidate, "profile": profile,
                  "status": "running", "spec_sha256": spec_hash, "study_sha256": self.contract_hash,
                  "workspace": str(workspace.resolve())}
        write_json(directory / "trial.json", record)
        try:
            snapshot = export_repository(self.service.root, profile, workspace, offline=True)
            self.adapter.prepare_dataset(self.service.root,profile,workspace,offline=True)
            manifest = self.adapter.materialize(workspace,profile,spec_hash)
            record.update(snapshot=snapshot, manifest=manifest)
            write_json(directory / "trial.json", record)
            execution = self.service.runner.run(workspace,self.adapter.steps(profile,split="validation"),env,on_event=self.emit)
            record["execution"] = execution
            check_execution(execution)
            record["validation"] = self.adapter.verify(profile,execution,workspace,snapshot,manifest,spec_hash,split="validation")
            record["status"] = "completed"
            record["metrics"] = record["validation"]["metrics_comparison"]["actual"]
            if time.monotonic() > self.deadline:
                raise TimeoutError("优化总时间预算耗尽")
        except (KeyboardInterrupt, SystemExit) as exc:
            record.update(status="interrupted", error="训练已中断")
            if getattr(exc, "execution", None) is not None:
                record["execution"] = exc.execution
            raise
        except TimeoutError as exc:
            record.update(status="timed_out", error=str(exc))
            raise
        except (ValueError, OSError, KeyError) as exc:
            record.update(status="failed", error=str(exc))
        finally:
            record["elapsed_s"] = time.monotonic()-start
            write_json(directory / "trial.json", record)
            self.runs[label] = record
        return record

    def holdout(self, label):
        if not self.selection_closed:
            raise ValueError("候选尚未冻结，禁止留出评估")
        self.check_contract()
        if self.remaining() <= 0:
            raise TimeoutError("留出确认预算耗尽")
        if self.remaining() <= 5:
            raise InsufficientBudget("留出确认预算不足")
        record = self.runs[label]
        if record["status"] != "completed":
            raise ValueError("失败训练不能进入留出确认")
        profile, workspace = record["profile"], Path(record["workspace"])
        env = {**profile["environment"], "require_prepared": True, "cache_lock_timeout_s": 0,
               "deadline_monotonic": min(self.deadline,time.monotonic()+120)}
        holdout = {"label": label, "status": "running"}
        write_json(workspace.parent / "holdout.json", holdout)
        try:
            execution = self.service.runner.run(workspace,[self.adapter.evaluation_step(profile,"holdout")],env,on_event=self.emit)
            holdout["execution"] = execution
            check_execution(execution)
            result = self.adapter.verify(profile,execution,workspace,record["snapshot"],record["manifest"],record["spec_sha256"],split="holdout")
            if time.monotonic() > self.deadline:
                raise TimeoutError("留出确认超过优化总时间预算")
            holdout.update(status="completed", validation=result)
            return result["metrics_comparison"]["actual"][self.metric]
        except (KeyboardInterrupt, SystemExit) as exc:
            holdout.update(status="interrupted", error="留出确认已中断")
            if getattr(exc, "execution", None) is not None:
                holdout["execution"] = exc.execution
            raise
        except Exception as exc:
            holdout.update(status="timed_out" if isinstance(exc, TimeoutError) else "failed", error=str(exc))
            raise
        finally:
            write_json(workspace.parent / "holdout.json", holdout)


@termination_signals()
def validate_candidates(service, profile, data, request, started, emit):
    with FileLock(str(Path(data["run_dir"]) / ".optimization.lock"), timeout=0):
        return _validate_candidates(service, profile, data, request, started, emit)


def _validate_candidates(service, profile, data, request, started, emit):
    max_candidates, budget = request.get("max_candidates",3), request.get("budget_seconds",7200)
    if isinstance(max_candidates,bool) or not isinstance(max_candidates,int) or not 1 <= max_candidates <= 3:
        raise ValueError("候选数量必须为1至3")
    if isinstance(budget,bool) or not isinstance(budget,(int,float)) or not math.isfinite(budget) or not 0 < budget <= 7200:
        raise ValueError("优化预算必须在0至7200秒之间")
    study = Study(service,profile,data["run_dir"],started,budget,emit)
    result = {"mode": "validate", "optimized": False, "available": True, "requested": True,
              "status": "running", "trials": [], "suggestions": [], "confirmation": [], "confirmation_training": [],
              "study_sha256": study.contract_hash, "baseline_run_id": Path(data["run_dir"]).name,
              "budget_seconds": budget, "max_candidates": max_candidates, "reason": "优化实验进行中，尚未完成候选确认"}
    def checkpoint(phase=None, label=None):
        if phase is not None:
            result.update(active_phase=phase, active_trial=label)
        result["budget_used_s"] = time.monotonic()-started
        write_json(Path(data["run_dir"]) / "optimization.json",result)
    def finish(status, reason):
        result.update(status=status,reason=reason)
        if status != "validated_gain":
            result["optimized"] = False
        for group in ("trials", "confirmation_training", "confirmation"):
            for row in result[group]:
                if row.get("status") == "running":
                    row["status"] = "timed_out" if status == "budget_exhausted" else status
        checkpoint()
        return result
    checkpoint("baseline", "baseline_2021")
    try:
        if study.remaining() <= 0:
            raise TimeoutError("优化总时间预算耗尽")
        if study.remaining() <= 2400:
            raise InsufficientBudget("剩余预算不足以预留40分钟确认阶段")
        baseline = study.train("baseline_2021")
        if baseline["status"] != "completed":
            return finish("failed","独立优化协议的实测基线失败")
        result["baseline"] = baseline["metrics"]
        checkpoint("advice")
        advice = suggest(service.llm,study.profile,baseline["validation"],data["method_sources"],
                         timeout_s=min(45,study.remaining()), study_contract=study.contract,
                         baseline_provenance={"baseline_run_id": Path(data["run_dir"]).name,
                            "trial_label": baseline["label"], "spec_sha256": baseline.get("spec_sha256"),
                            "study_sha256": study.contract_hash})
        result.update({key: advice[key] for key in ("suggestions","calls","model","usage","advice_context","advice_history") if key in advice})
        checkpoint()
        if advice["status"] != "suggested":
            return finish(advice["status"],advice["reason"])
        best = None
        baseline_score = baseline["metrics"][study.metric]
        for index, candidate in enumerate(result["suggestions"][:max_candidates],1):
            if study.remaining() <= 0:
                raise TimeoutError("优化总时间预算耗尽")
            if study.remaining() <= 2400:
                break
            # Every attempted candidate counts, including failures.
            label = f"candidate_{index}_2021"
            summary = {"candidate": index, "parameter": candidate["parameter"], "value": candidate["value"],
                       "run": f"trials/{label}", "status": "running"}
            result["trials"].append(summary)
            checkpoint("candidate_training", label)
            trial = study.train(label,candidate=candidate,reserve_s=2400)
            candidate["status"] = "tested" if trial["status"] == "completed" else "failed"
            summary.update(status=trial["status"], metrics=trial.get("metrics"),
                           elapsed_s=trial["elapsed_s"], error=trial.get("error",""))
            if trial["status"] == "completed":
                score = trial["metrics"][study.metric]
                if study.direction*(score-baseline_score)>0 and (best is None or study.direction*(score-best["metrics"][study.metric])>0):
                    best = trial
            checkpoint()
        if best is None:
            completed = any(t["status"] == "completed" for t in result["trials"])
            return finish("tested_no_gain" if completed else "budget_insufficient" if not result["trials"] else "failed",
                          "没有候选在固定验证集超过实测基线，未使用留出集选优" if completed else "没有完成有效候选比较；保留失败记录")
        study.selection_closed = True
        result["selected_candidate"] = best["candidate"]
        write_json(Path(data["run_dir"]) / "selection.json", {"study_sha256": study.contract_hash,
                   "selected_run": best["label"], "metric": study.metric, "validation_value": best["metrics"][study.metric]})
        for label,candidate in [("baseline_2022",None),("candidate_2022",best["candidate"])]:
            summary = {"label": label, "status": "running"}
            result["confirmation_training"].append(summary)
            checkpoint("confirmation_training", label)
            trained = study.train(label,seed=2022,candidate=candidate)
            summary.update(status=trained["status"], metrics=trained.get("metrics"))
            checkpoint()
            if trained["status"] != "completed":
                return finish("confirmation_failed","第二种子完整训练失败；没有已验证提升")
        for seed,base_label,best_label in [(2021,"baseline_2021",best["label"]),(2022,"baseline_2022","candidate_2022")]:
            confirmation = {"seed": seed,"metric": study.metric,"status": "running"}
            result["confirmation"].append(confirmation)
            checkpoint("holdout", base_label)
            base_value = study.holdout(base_label)
            confirmation["baseline"] = base_value
            checkpoint("holdout", best_label)
            candidate_value = study.holdout(best_label)
            passed = confirmed_gain(profile["adapter_id"],base_value,candidate_value)
            confirmation.update(status="completed", candidate=candidate_value, **{"pass": passed})
            checkpoint()
        result["optimized"] = all(row["pass"] for row in result["confirmation"])
        if result["optimized"]:
            best["candidate"]["status"] = "validated_gain"
        return finish("validated_gain" if result["optimized"] else "tested_no_gain",
                      "两种子留出结果均达到事前门槛" if result["optimized"] else "验证集候选未在两种子留出确认中达到事前门槛")
    except (KeyboardInterrupt, SystemExit):
        for group in ("trials", "confirmation_training", "confirmation"):
            for row in result[group]:
                if row.get("status") == "running":
                    row["status"] = "interrupted"
        finish("interrupted","实验已中断，保留已完成候选及确认记录，不宣称已验证提升")
        raise
    except InsufficientBudget as exc:
        return finish("budget_insufficient",str(exc))
    except TimeoutError:
        for group in ("trials", "confirmation_training", "confirmation"):
            for row in result[group]:
                if row.get("status") == "running":
                    row["status"] = "timed_out"
        return finish("budget_exhausted","预算耗尽，保留已完成记录，不宣称已验证提升")
    except (ValueError, OSError, KeyError) as exc:
        return finish("failed",str(exc))

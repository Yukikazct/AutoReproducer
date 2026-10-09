"""Small real parameter studies. Selection never consumes holdout results."""
from copy import deepcopy
import math
import time
from pathlib import Path

from src.method_adapters import write_json, digest, RUNTIMES
from src.method_advice import suggest
from src.repository_adapters import get_adapter


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
        if self.remaining() <= reserve_s + 5:
            raise TimeoutError("优化总时间预算耗尽")
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
        snapshot = export_repository(self.service.root, profile, workspace, offline=True)
        self.adapter.prepare_dataset(self.service.root,profile,workspace,offline=True)
        manifest = self.adapter.materialize(workspace,profile,spec_hash)
        env = {**profile["environment"], "require_prepared": True, "cache_lock_timeout_s": 0,
               "deadline_monotonic": min(self.deadline-reserve_s, start+1200)}
        record = {"label": label, "seed": seed, "candidate": candidate, "profile": profile,
                  "spec_sha256": spec_hash, "study_sha256": self.contract_hash, "snapshot": snapshot,
                  "manifest": manifest, "workspace": str(workspace.resolve()), "status": "running"}
        write_json(directory / "trial.json", record)
        self.runs[label] = record
        try:
            execution = self.service.runner.run(workspace,self.adapter.steps(profile,split="validation"),env,on_event=self.emit)
            record["execution"] = execution
            record["validation"] = self.adapter.verify(profile,execution,workspace,snapshot,manifest,spec_hash,split="validation")
            record["status"] = "completed"
            record["metrics"] = record["validation"]["metrics_comparison"]["actual"]
        except KeyboardInterrupt as exc:
            record.update(status="interrupted", error="实验已中断", elapsed_s=time.monotonic()-start)
            if getattr(exc, "execution", None):
                record["execution"] = exc.execution
            write_json(directory / "trial.json", record)
            raise
        except (ValueError, OSError, KeyError) as exc:
            record.update(status="failed", error=str(exc))
        record["elapsed_s"] = time.monotonic()-start
        if time.monotonic() > self.deadline:
            record.update(status="failed",error="优化总时间预算耗尽")
        write_json(directory / "trial.json", record)
        self.runs[label] = record
        return record

    def holdout(self, label):
        if not self.selection_closed:
            raise ValueError("候选尚未冻结，禁止留出评估")
        self.check_contract()
        if self.remaining() <= 5:
            raise TimeoutError("留出确认预算不足")
        record = self.runs[label]
        if record["status"] != "completed":
            raise ValueError("失败训练不能进入留出确认")
        profile, workspace = record["profile"], Path(record["workspace"])
        env = {**profile["environment"], "require_prepared": True, "cache_lock_timeout_s": 0,
               "deadline_monotonic": min(self.deadline,time.monotonic()+120)}
        evidence = {"status": "running", "label": label, "study_sha256": self.contract_hash}
        path = workspace.parent / "holdout.json"
        write_json(path, evidence)
        try:
            execution = self.service.runner.run(workspace,[self.adapter.evaluation_step(profile,"holdout")],env,on_event=self.emit)
            evidence["execution"] = execution
            result = self.adapter.verify(profile,execution,workspace,record["snapshot"],record["manifest"],record["spec_sha256"],split="holdout")
            evidence.update(status="completed", validation=result)
        except KeyboardInterrupt as exc:
            evidence.update(status="interrupted", error="留出确认已中断")
            if getattr(exc, "execution", None):
                evidence["execution"] = exc.execution
            raise
        except (ValueError, OSError, KeyError) as exc:
            evidence.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(path, evidence)
        return result["metrics_comparison"]["actual"][self.metric]


def validate_candidates(service, profile, data, request, started, emit):
    max_candidates, budget = request.get("max_candidates",3), request.get("budget_seconds",7200)
    if isinstance(max_candidates,bool) or not isinstance(max_candidates,int) or not 1 <= max_candidates <= 3:
        raise ValueError("候选数量必须为1至3")
    if isinstance(budget,bool) or not isinstance(budget,(int,float)) or not math.isfinite(budget) or not 0 < budget <= 7200:
        raise ValueError("优化预算必须在0至7200秒之间")
    study = Study(service,profile,data["run_dir"],started,budget,emit)
    result = {"mode": "validate", "optimized": False, "available": True, "requested": True,
              "status": "running", "trials": [], "suggestions": [], "confirmation": [],
              "study_sha256": study.contract_hash, "baseline_run_id": Path(data["run_dir"]).name,
              "budget_seconds": budget, "max_candidates": max_candidates, "reason": "优化实验运行中"}
    # The caller retains the same object even when cancellation unwinds this stack.
    data["optimization"] = result
    def save(phase=None):
        if phase:
            result["phase"] = phase
        result["budget_used_s"] = time.monotonic()-started
        write_json(Path(data["run_dir"]) / "optimization.json", result)
    def finish(status, reason):
        result.update(status=status,reason=reason,budget_used_s=time.monotonic()-started)
        save()
        return result
    save("baseline")
    try:
        if study.remaining() <= 0:
            return finish("budget_exhausted", "优化总时间预算已经耗尽")
        if study.remaining() <= 2400:
            return finish("insufficient_budget","剩余预算不足以预留40分钟确认阶段")
        baseline = study.train("baseline_2021")
        if baseline["status"] != "completed":
            return finish("failed","独立优化协议的实测基线失败")
        result["baseline"] = baseline["metrics"]
        save("advice")
        advice = suggest(service.llm,study.profile,baseline["validation"],data["method_sources"],
                         timeout_s=min(45,study.remaining()), study_spec=study.contract,
                         baseline_run_id=f"{Path(data['run_dir']).name}/trials/baseline_2021")
        write_json(Path(data["run_dir"]) / "advice.json", advice)
        result["advice_path"] = "advice.json"
        result.update({key: advice[key] for key in ("suggestions","calls","model","usage") if key in advice})
        if advice["status"] != "suggested":
            return finish(advice["status"],advice["reason"])
        best = None
        baseline_score = baseline["metrics"][study.metric]
        for index, candidate in enumerate(result["suggestions"][:max_candidates],1):
            if study.remaining() <= 2400:
                break
            # Every attempted candidate counts, including failures.
            label = f"candidate_{index}_2021"
            result["active_trial"] = label
            save("candidate_selection")
            trial = study.train(label,candidate=candidate,reserve_s=2400)
            candidate["status"] = "tested" if trial["status"] == "completed" else "failed"
            result["trials"].append({"candidate": index, "parameter": candidate["parameter"], "value": candidate["value"],
                                     "run": f"trials/{label}", "status": trial["status"], "metrics": trial.get("metrics"),
                                     "elapsed_s": trial["elapsed_s"], "error": trial.get("error","")})
            if trial["status"] == "completed":
                score = trial["metrics"][study.metric]
                if study.direction*(score-baseline_score)>0 and (best is None or study.direction*(score-best["metrics"][study.metric])>0):
                    best = trial
            save()
        if best is None:
            completed = any(t["status"] == "completed" for t in result["trials"])
            return finish("tested_no_gain" if completed else "insufficient_budget" if not result["trials"] else "failed",
                          "没有候选在固定验证集超过实测基线，未使用留出集选优" if completed else "没有完成有效候选比较；保留失败记录")
        study.selection_closed = True
        result["selected_candidate"] = best["candidate"]
        write_json(Path(data["run_dir"]) / "selection.json", {"study_sha256": study.contract_hash,
                   "selected_run": best["label"], "metric": study.metric, "validation_value": best["metrics"][study.metric]})
        for label,candidate in [("baseline_2022",None),("candidate_2022",best["candidate"])]:
            result["active_trial"] = label
            save("confirmation_training")
            trained = study.train(label,seed=2022,candidate=candidate)
            if trained["status"] != "completed":
                return finish("confirmation_failed","第二种子完整训练失败；没有已验证提升")
        for seed,base_label,best_label in [(2021,"baseline_2021",best["label"]),(2022,"baseline_2022","candidate_2022")]:
            result["active_trial"] = base_label
            save("holdout_confirmation")
            base_value = study.holdout(base_label)
            result["active_trial"] = best_label
            save()
            candidate_value = study.holdout(best_label)
            passed = confirmed_gain(profile["adapter_id"],base_value,candidate_value)
            result["confirmation"].append({"seed": seed,"metric": study.metric,"baseline": base_value,
                                           "candidate": candidate_value,"pass": passed})
            save()
        result["optimized"] = all(row["pass"] for row in result["confirmation"])
        if result["optimized"]:
            best["candidate"]["status"] = "validated_gain"
        return finish("validated_gain" if result["optimized"] else "tested_no_gain",
                      "两种子留出结果均达到事前门槛" if result["optimized"] else "验证集候选未在两种子留出确认中达到事前门槛")
    except KeyboardInterrupt:
        result["optimized"] = False
        finish("interrupted", "实验已中断；保留已完成试验与确认记录，不宣称已验证提升")
        raise
    except TimeoutError:
        return finish("budget_exhausted" if study.remaining() <= 0 else "insufficient_budget",
                      "剩余预算不足以继续，保留已完成记录，不宣称已验证提升")
    except (ValueError, OSError, KeyError) as exc:
        return finish("failed",str(exc))

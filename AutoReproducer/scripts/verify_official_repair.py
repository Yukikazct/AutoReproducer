"""Real API + Docker fault injection; never alter the official source cache.

Run with LLM_API_KEY in the environment, or --ask-key for hidden input.
This deliberately introduces an invalid config attribute in the execution copy,
then requires an actual LLM patch and successful ETTh1 training/evaluation.
No Markdown report is automatically saved.
"""
import argparse
import getpass
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator

TITLE = "Are Transformers Effective for Time Series Forecasting?"
TARGET = "main/models/DLinear.py"
ORIGINAL = "self.seq_len = configs.seq_len"
FAULT = "self.seq_len = configs.sequence_length"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask-key", action="store_true", help="交互式隐藏输入 API key")
    args = parser.parse_args(argv)
    key = getpass.getpass("API key (hidden): ") if args.ask_key else os.environ.get("LLM_API_KEY", "")
    if not key:
        parser.error("设置 LLM_API_KEY 或使用 --ask-key")
    available, reason = BaseAgent.docker_engine_available()
    if not available:
        parser.error(f"Docker 不可用：{reason}")
    llm = LLMClient(base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
                    model=os.environ.get("LLM_MODEL", "deepseek-chat"), api_key=key, timeout=300)
    orch = Orchestrator(llm_client=llm, mock_mode=False, use_docker=True, max_trials=0,
                        progress_cb=lambda state, agent, status: print(f"[{state}] {agent} {status}", flush=True))
    executor = orch.agents["executor"]
    original_step = executor._execute_plan_step
    original_plan = executor._execute_plan
    injection = {"kind": "fault_injection", "path": TARGET,
                 "description": "仅临时执行副本：将 seq_len 配置属性替换为不存在的 sequence_length；非上游原生错误。",
                 "old": ORIGINAL, "new": FAULT}
    cached_path, cache_before, injected = None, None, False

    def execute_step(step, workspace, units):
        nonlocal cached_path, cache_before, injected
        if step.kind == "run" and not injected:
            cached_path = Path(orch.data["storage"]["fetched"]["code"]["path"]) / "models/DLinear.py"
            cache_before = hashlib.sha256(cached_path.read_bytes()).hexdigest()
            path = Path(workspace) / TARGET
            source = path.read_text(encoding="utf-8")
            if source.count(ORIGINAL) != 1:
                raise ValueError("故障注入目标不匹配，停止验证")
            path.write_text(source.replace(ORIGINAL, FAULT, 1), encoding="utf-8")
            injection["injected_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            injected = True
            print("[FAULT_INJECTION] 临时副本已注入配置属性错误；官方缓存未改动", flush=True)
        return original_step(step, workspace, units)

    def execute_plan(plan, paper_info):
        result = original_plan(plan, paper_info)
        result["verification_scenario"] = dict(injection)
        if cached_path is not None:
            result["verification_scenario"]["cache_unchanged"] = (
                hashlib.sha256(cached_path.read_bytes()).hexdigest() == cache_before)
        return result

    executor._execute_plan_step = execute_step
    executor._execute_plan = execute_plan
    result = orch.run({"paper_title": TITLE, "use_llm_pipeline": True})
    data = result["data"]
    execution = data.get("execution") or {}
    metrics = execution.get("actual_metrics") or {}
    repairs = [r for stage in execution.get("stages", []) for r in stage.get("repairs", [])]
    passed = (result["state"] == "COMPLETED"
              and (data.get("validation") or {}).get("status") == "smoke_verified"
              and execution.get("source_modified")
              and any(r.get("source") == "llm" and r.get("status") == "verified" and r.get("patches") for r in repairs)
              and execution.get("verification_scenario", {}).get("cache_unchanged")
              and all(isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k]) for k in ("mse", "mae")))
    print(json.dumps({"verification": "fault_injection", "passed": bool(passed), "state": result["state"],
                      "error": result.get("error"), "metrics": metrics, "llm_calls": llm.call_count,
                      "repair_rounds": execution.get("llm_repair_rounds"), "evidence_dir": execution.get("evidence_dir"),
                      "cache_unchanged": execution.get("verification_scenario", {}).get("cache_unchanged")},
                     ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Run a fixed real-paper experiment, optionally checking its public protocol via API."""
import argparse
import getpass
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator import Orchestrator
from src.repository_profiles import PROFILE_LABELS
from src.llm.llm_client import LLMClient
from frontend.llm_config import resolve_llm_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=list(PROFILE_LABELS), default="dlinear_etth1_reference")
    parser.add_argument("--prepare-only", action="store_true", help="只校验代码、数据并保存计划，不安装依赖、不训练")
    parser.add_argument("--prepare-environment", action="store_true", help="方法预设：准备代码、数据、依赖和设备，不训练")
    parser.add_argument("--optimization", choices=["off", "suggest", "validate"], default="off")
    parser.add_argument("--max-candidates", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--budget-seconds", type=int, default=7200, help="优化训练与评估总预算，最多7200秒")
    parser.add_argument("--offline", action="store_true", help="准备时只复用经版本/哈希校验的本地代码与数据；缺失即失败")
    parser.add_argument("--llm-review", action="store_true", help="用真实API执行有公开来源依据的多Agent分析；密钥从环境或隐藏输入读取")
    parser.add_argument("--analysis-mode", choices=["multi_agent", "public_protocol"], default="multi_agent")
    parser.add_argument("--result-review", action="store_true", help="允许API分析本次指标、训练轮数和核验状态摘要；不发送路径、原始日志或文件")
    args = parser.parse_args(argv)
    if args.prepare_only and args.prepare_environment:
        parser.error("--prepare-only and --prepare-environment are mutually exclusive")
    if not 1 <= args.budget_seconds <= 7200:
        parser.error("--budget-seconds must be between 1 and 7200")
    if args.prepare_environment and args.profile.startswith("dlinear"):
        parser.error("--prepare-environment is available for method experiment profiles")
    if args.optimization != "off" and args.profile.startswith("dlinear"):
        parser.error("DLinear preserves the author protocol; parameter optimization is available for SIREN and Neural ODE")
    if args.result_review and not args.llm_review:
        parser.error("--result-review requires --llm-review")

    def progress(event):
        if event.get("type") == "state":
            print(f"[{event['state']}] {event.get('agent', '')}: {event.get('status', '')}", flush=True)
        elif event.get("type") == "repository_output":
            print(event.get("text", ""), end="", flush=True)

    llm = None
    use_llm_review = args.llm_review and not (args.prepare_only or args.prepare_environment)
    if use_llm_review or (args.optimization != "off" and not (args.prepare_only or args.prepare_environment)):
        token = os.environ.get("LLM_API_KEY") or (getpass.getpass("API Key (hidden): ") if sys.stdin.isatty() else "")
        cfg = resolve_llm_config(api_key=token)
        llm = LLMClient(**cfg, mock_mode=False, timeout=120, max_tokens=8192)
        del token, cfg
    result = Orchestrator(mock_mode=False, llm_client=llm).run({"experiment_profile": args.profile,
        "prepare_only": args.prepare_only, "offline": args.offline,
        **({"prepare_environment": args.prepare_environment, "optimization_mode": args.optimization,
            "max_candidates": args.max_candidates, "budget_seconds": args.budget_seconds}
           if not args.profile.startswith("dlinear") else {}),
        "use_llm_review": use_llm_review, "analysis_mode": args.analysis_mode,
        "allow_result_summary_review": args.result_review and use_llm_review}, on_event=progress)
    data = result["data"]
    print(json.dumps({"state": result["state"], "error": result.get("error"),
                      "validation": data.get("validation"), "run_dir": data.get("run_dir")},
                     ensure_ascii=False, indent=2))
    return 0 if result["state"] == "COMPLETED" and (
        data.get("validation") or {}).get("result_level") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

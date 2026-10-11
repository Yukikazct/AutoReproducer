"""Run a fixed real-paper experiment, optionally checking its public protocol via API."""
import argparse
import getpass
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.repository_profiles import PROFILE_LABELS
from src.llm.llm_client import LLMClient
from frontend.llm_config import resolve_llm_config
from src.process_lifecycle import termination_signals, watch_parent_session


def Orchestrator(*args, **kwargs):
    # A bare compatible Python can bootstrap before importing app dependencies.
    from src.orchestrator import Orchestrator as implementation
    return implementation(*args, **kwargs)


@watch_parent_session()
@termination_signals()
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

    # Resolve hidden input before handing control to a noninteractive worker.
    # Python older than 3.11 can still bootstrap without tomllib.
    try:
        from src.local_llm_settings import load_local_llm_settings
    except ModuleNotFoundError as exc:
        if exc.name != "tomllib":
            raise
    else:
        load_local_llm_settings()
    use_llm_review = args.llm_review and not (args.prepare_only or args.prepare_environment)
    needs_llm = use_llm_review or (args.optimization != "off" and not (args.prepare_only or args.prepare_environment))
    token = ""
    if needs_llm:
        token = os.environ.get("LLM_API_KEY") or (getpass.getpass("API Key (hidden): ") if sys.stdin.isatty() else "")

    from src.runtime_preparation import runtime_requirement, prepare_runtime, run_owned_process, safe_runtime_diagnostic
    from src.runtime_preparation import PRESET_WORKER_PREPARATION_ALLOWANCE_S
    reason = runtime_requirement()
    if reason:
        print(reason, flush=True)
        prepared = prepare_runtime(Path(__file__).resolve().parents[1], offline=args.offline,
            progress_callback=lambda event: print(event.get("message", ""), flush=True))
        forwarded = list(argv) if argv is not None else sys.argv[1:]
        worker_env = os.environ.copy()
        if token:
            worker_env["LLM_API_KEY"] = token
        print("安全运行环境已就绪，正在执行预设；结束后输出完整结果。", flush=True)
        completed = run_owned_process(
            [prepared.executable, "-X", "utf8", str(Path(__file__).resolve()), *forwarded],
            cwd=Path(__file__).resolve().parents[1],
            timeout_s=max(7200, args.budget_seconds) + PRESET_WORKER_PREPARATION_ALLOWANCE_S,
            env=worker_env,
        )
        def private_output(value):
            for name in ("LLM_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "AUTOREPRO_GITHUB_TOKEN"):
                if worker_env.get(name):
                    value = value.replace(worker_env[name], "[REDACTED]")
            return value
        if completed.stdout:
            print(private_output(completed.stdout), end="", flush=True)
        if completed.stderr:
            print(safe_runtime_diagnostic(private_output(completed.stderr)), end="", file=sys.stderr, flush=True)
        return completed.returncode

    def progress(event):
        if event.get("type") == "state":
            print(f"[{event['state']}] {event.get('agent', '')}: {event.get('status', '')}", flush=True)
        elif event.get("type") == "repository_output":
            print(event.get("text", ""), end="", flush=True)

    llm = None
    if needs_llm:
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
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("实验已中断，已有记录已保存。", file=sys.stderr, flush=True)
        raise SystemExit(130)

"""real_e2e.py - 真实模式端到端：跑完整流水线并打印关键结论。

与 Mock 模式测试的区别：走真实 LLM（OpenAI 兼容接口）+ 真实子进程执行，
因此能暴露 Mock 永远掩盖不到的问题——路径拼接、pip 参数、指标键大小写、
模型实际生成质量、官方入口脚本能不能真的跑起来。本仓库最近几处真实模式
缺陷都是靠这条链路发现的。

已适配标题默认使用官方 CPU 冒烟配置，无需 LLM；添加 `--llm-pipeline`
可强制测试 API 官方 CPU 流水线，当前支持 DLinear / ETTh1。

三种输入方式（任选其一）：

    # 1) 样例 / 本地 PDF
    python scripts/real_e2e.py samples/paper/minimal_linear_regression.pdf

    # 2) 只有标题（无 PDF 时也能跑：论文信息由标题推断）
    python scripts/real_e2e.py --paper-title "iTransformer: Inverted \
Transformers Are Effective for Time Series Forecasting"

    # 3) 直接指定官方仓库，跳过代码检索的不确定性
    python scripts/real_e2e.py --paper-title "iTransformer" \
        --repo-url https://github.com/thuml/iTransformer

API key **只从环境变量读取，不落盘**：

    $env:LLM_API_KEY = "sk-..."          # PowerShell
    export LLM_API_KEY="sk-..."          # bash/zsh

可选环境变量：`LLM_BASE_URL`（默认 https://api.deepseek.com）、
`LLM_MODEL`（默认 deepseek-chat）。

**关于 `--use-docker`**：官方代码必须在加固 Docker 沙箱里跑。
预设与 `--llm-pipeline` 都自动启用 Docker 并检查引擎，失败保留官方执行
记录与报告。通用 PDF 路径需显式添加此选项。先启动 Docker：

    python scripts/real_e2e.py --paper-title "iTransformer" --use-docker

跑完在终端打印各阶段关键结论。默认不保存报告；添加 `--save-report`
保存到 `data/reports/_real_e2e_report.md`，或用 `--report-out` 指定保存路径。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_client import LLMClient       # noqa: E402
from src.orchestrator import Orchestrator      # noqa: E402

DEFAULT_PDF = "samples/paper/minimal_linear_regression.pdf"
WORKSPACE = "data/_e2e_ws"     # 通用生成代码的真实优化工作区
REPORT_OUT = "data/reports/_real_e2e_report.md"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="真实模式端到端复现流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdf", nargs="?", default="",
                   help=f"论文 PDF 路径（默认 {DEFAULT_PDF}）")
    p.add_argument("--paper-title", default="",
                   help="只给论文标题（无 PDF 时由标题推断论文信息）")
    from src.experiment_profiles import PROFILES
    p.add_argument("--profile", choices=list(PROFILES), default="",
                   help="固定官方 CPU 真实冒烟预设（自动启用 Docker，无需 LLM API Key）")
    p.add_argument("--repo-url", default="",
                   help="已知官方仓库 URL，跳过代码检索直接使用")
    p.add_argument("--use-docker", action="store_true",
                   help="启用加固 Docker 沙箱执行官方代码（需 Docker 已启动）")
    p.add_argument("--llm-pipeline", action="store_true",
                   help="真实 API 读取官方仓库并规划 CPU 实验（当前 DLinear / ETTh1，自动启用 Docker）")
    p.add_argument("--workspace", default=WORKSPACE,
                   help=f"优化工作区（默认 {WORKSPACE}）")
    p.add_argument("--max-trials", type=int, default=2,
                   help="优化试验上限（默认 2）")
    p.add_argument("--save-report", action="store_true", help=f"保存报告到 {REPORT_OUT}")
    p.add_argument("--report-out", default="",
                   help="明确选择保存报告并指定路径（默认不保存）")
    args = p.parse_args(argv)
    if not args.pdf and not args.paper_title and not args.profile:
        args.pdf = DEFAULT_PDF          # 都没给 -> 退回内置样例
    return args


def _print_plan(data):
    """计划模式（多代码单元）的可见性输出。"""
    print()
    print("--- 2. 执行计划（多代码单元）---")
    ex = data.get("execution", {}) or {}
    mode = ex.get("execution_mode") or "generated"
    print("  execution_mode:", mode)
    # 回退时顶层 execution 是生成路径的结果，官方执行记录在 plan_execution 里
    src = ex.get("plan_execution") or ex
    plan = src.get("plan") or {}
    if not plan:
        print("  (无计划：官方准备失败，已停止执行)" if ex.get("execution_mode") == "plan"
              else "  (无计划：未发现可用官方代码，走了生成脚本路径)")
        return

    for u in plan.get("units", []) or []:
        print(f"  unit {u.get('unit_id')}: [{u.get('role')}] "
              f"{u.get('fetch_state')} <- {u.get('url')}")
    entry = plan.get("entry") or {}
    print("  入口:", entry.get("script"),
          f"(interp={entry.get('interp')}, unit={entry.get('unit_id')})")
    for note in plan.get("notes", []) or []:
        print("  备注:", str(note)[:160])

    print("  步骤:")
    stages = {s.get("step_id"): s for s in (src.get("stages") or [])}
    for st in plan.get("steps", []) or []:
        rec = stages.get(st.get("step_id")) or {}
        if rec.get("skipped_deps"):
            mark = "⏭️ 跳过（前置失败）"
        elif rec.get("success"):
            mark = "✅"
        elif rec == {}:
            mark = "—"                   # 计划里有、执行没到（前序中断）
        else:
            mark = f"❌ 退出码 {rec.get('exit_code')}"
        print(f"    [{st.get('kind')}] {st.get('step_id')}: {mark}")
        print(f"        {str(st.get('cmd'))[:150]}")
        for rp in rec.get("repairs", []) or []:
            print(f"        修复: {rp.get('error_type')} -> "
                  f"{rp.get('strategy')} ({str(rp.get('detail'))[:80]})")

    metrics = src.get("actual_metrics") or {}
    print("  实际指标:", metrics or "(未提取到)")
    if src.get("plan_failed_irreparably"):
        print("  计划不可修复失败 ->", str(src.get("plan_fail_reason"))[:200])


def _print_generated(data):
    """生成脚本路径的可见性输出（含回退原因）。"""
    ex = data.get("execution", {}) or {}
    print()
    print("--- 3. 官方脚本执行输出 ---" if ex.get("execution_mode") == "plan"
          else "--- 3. 生成脚本执行 ---")
    if ex.get("execution_mode") == "generated_fallback":
        print("  ⚠ 官方计划失败后回退:", str(ex.get("plan_fail_reason"))[:200])
    code = ex.get("code", "") or ""
    print("  代码:", len(code), "字符 /", len(code.splitlines()), "行")
    print("  sanitize_stats:", ex.get("sanitize_stats"))
    print("  not_runnable:", ex.get("not_runnable"),
          "| best_effort:", ex.get("best_effort"),
          "| fallback_used:", ex.get("fallback_used"))
    print("  末行:", repr(code.rstrip().splitlines()[-1] if code.strip() else ""))
    final = ex.get("final") or {}
    print("  exit_code:", final.get("exit_code"))
    print("  stdout 尾部:")
    for ln in (final.get("stdout") or "").strip().splitlines()[-12:]:
        print("   ", ln)


def main(argv=None):
    args = _parse_args(argv)

    from src.experiment_profiles import resolve_profile
    if args.llm_pipeline and args.profile:
        sys.exit("--llm-pipeline 与 --profile 不能同时使用")
    args.profile = "" if args.llm_pipeline else resolve_profile(args.paper_title, args.profile)

    key = os.environ.get("LLM_API_KEY", "")
    if not key and not args.profile:
        sys.exit("未设置 LLM_API_KEY（本脚本不会从文件读取密钥）")
    if args.profile or args.llm_pipeline:
        from src.base_agent import BaseAgent
        available, reason = BaseAgent.docker_engine_available()
        if not available:
            sys.exit(f"请启动 Docker Desktop 后重试：{reason}")
        args.use_docker = True

    os.makedirs(args.workspace, exist_ok=True)

    llm = LLMClient(base_url=os.environ.get("LLM_BASE_URL",
                                            "https://api.deepseek.com"),
                    model=os.environ.get("LLM_MODEL", "deepseek-chat"),
                    api_key=key, mock_mode=False, timeout=300)
    def progress(state, agent, status):
        if status in ("running", "success", "error"):
            print(f"[{state}] {agent} {status}", flush=True)
        if state == "FIND_RESOURCES" and status == "success":
            resources = orch.data.get("resources") or {}
            discovery = resources.get("repo_discovery") or {}
            print("  仓库:", resources.get("code_repo_url", "未找到"), flush=True)
            print("  发现路径:", " → ".join(discovery.get("discovery_chain") or []), flush=True)

    orch = Orchestrator(llm_client=llm, mock_mode=False,
                        use_docker=args.use_docker,
                        max_trials=args.max_trials,
                        workspace_dir=args.workspace, progress_cb=progress)

    payload = {}
    if args.llm_pipeline:
        payload["use_llm_pipeline"] = True
    if args.profile:
        payload["experiment_profile"] = args.profile
    if args.paper_title:
        payload["paper_title"] = args.paper_title
    if args.pdf:
        payload["pdf_path"] = args.pdf
    if args.repo_url:
        payload["preferred_repo_url"] = args.repo_url

    result = orch.run(payload)
    data = result["data"]
    if result.get("error"):
        print("实验错误:", result["error"])

    print("=" * 60)
    print("输入:", payload)
    print("沙箱:", "docker（官方代码可执行）" if args.use_docker
          else "本地隔离（官方代码会被拒绝执行）")
    print("最终状态:", result["state"], "| error:", result.get("error"))
    print("审计统计:", {k: v for k, v in (result.get("audit_stats") or {}).items()
                        if k in ("total_steps", "success", "errors",
                                 "duration_sec", "llm_calls")})

    print()
    print("--- 1. PaperReader ---")
    pi = data.get("paper_info", {})
    for k in ("title", "method", "dataset", "metrics", "info_sufficient"):
        v = str(pi.get(k, ""))
        print(f"  {k}: {v[:140]}")

    _print_plan(data)
    _print_generated(data)

    print()
    print("--- 4. ResultValidator ---")
    va = data.get("validation", {}) or {}
    print("  status:", va.get("status"))
    print("  is_reproduced:", va.get("is_reproduced"))
    print("  confidence:", va.get("confidence"))
    print("  reason:", str(va.get("reason"))[:200])
    mc = va.get("metrics_comparison", {}) or {}
    print("  论文声明:", mc.get("paper"))
    print("  实际运行:", mc.get("actual"))

    print()
    print("--- 5. 优化 ---")
    op = data.get("optimization", {}) or {}
    print("  optimized:", op.get("optimized"), "|", str(op.get("reason"))[:100])
    print("  基线:", op.get("baseline"), "| 最优:", op.get("best_result"),
          "| 幅度:", op.get("improvement"))
    for r in op.get("optimization_report") or []:
        d = r.get("detail") or {}
        print(f"  - [{d.get('type')}] {str(r.get('arm'))[:44]}")
        print(f"      improvement={r.get('improvement')} kept={r.get('kept')} "
              f"metric={d.get('metric')} basis={d.get('reward_basis')}")
        print(f"      status={d.get('status')} reason={d.get('reason')}")

    print()
    print("--- 6. 报告（验证结论段）---")
    report = data.get("report", "") or ""
    out, keep = [], False
    for ln in report.splitlines():
        if ln.startswith("## 5."):
            keep = True
        elif ln.startswith("## 7."):
            break
        if keep:
            out.append(ln)
    print("\n".join(out or ["(未找到第 5 节)"]))

    if args.report_out or args.save_report:
        out_path = Path(args.report_out or REPORT_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n完整报告已写入 {out_path}")
    else:
        print("\n报告未保存；需要保留时请使用 --save-report 或 --report-out。")
    if args.profile:
        return 0 if va.get("status") == "smoke_verified" else 1
    if args.llm_pipeline:
        return 0 if result.get("state") == "COMPLETED" and va.get("status") == "smoke_verified" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

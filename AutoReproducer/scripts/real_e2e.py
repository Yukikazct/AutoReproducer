"""real_e2e.py - 真实模式端到端：用样例论文 PDF 跑完整流水线。

与 Mock 模式测试的区别：走真实 LLM（OpenAI 兼容接口）+ 真实子进程执行，
因此能暴露 Mock 永远掩盖不到的问题——路径拼接、pip 参数、指标键大小写、
模型实际生成质量等。本仓库最近几处真实模式缺陷都是靠这条链路发现的。

API key **只从环境变量读取，不落盘**：

    $env:LLM_API_KEY = "sk-..."          # PowerShell
    python scripts/real_e2e.py

可选：把论文 PDF 路径作为第一个参数（默认用内置样例）。

跑完在终端打印各阶段关键结论，并把完整报告写入
`data/reports/_real_e2e_report.md`。

用法:
    python scripts/real_e2e.py [论文.pdf]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_client import LLMClient       # noqa: E402
from src.orchestrator import Orchestrator      # noqa: E402

DEFAULT_PDF = "samples/paper/minimal_linear_regression.pdf"
WORKSPACE = "data/_e2e_ws"     # 传工作区 -> 走真实优化闭环（否则跑哈希模拟）
REPORT_OUT = "data/reports/_real_e2e_report.md"

key = os.environ.get("LLM_API_KEY", "")
if not key:
    sys.exit("未设置 LLM_API_KEY（本脚本不会从文件读取密钥）")

pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
os.makedirs(WORKSPACE, exist_ok=True)

llm = LLMClient(base_url="https://api.deepseek.com", model="deepseek-chat",
                api_key=key, mock_mode=False, timeout=300)
orch = Orchestrator(llm_client=llm, mock_mode=False, use_docker=False,
                    max_trials=2, workspace_dir=WORKSPACE)

result = orch.run({"pdf_path": pdf})
data = result["data"]

print("=" * 60)
print("论文:", pdf)
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

print()
print("--- 2. 代码执行 ---")
ex = data.get("execution", {}) or {}
code = ex.get("code", "") or ""
print("  代码:", len(code), "字符 /", len(code.splitlines()), "行")
print("  sanitize_stats:", ex.get("sanitize_stats"))
print("  not_runnable:", ex.get("not_runnable"))
print("  best_effort:", ex.get("best_effort"),
      "| fallback_used:", ex.get("fallback_used"))
print("  末行:", repr(code.rstrip().splitlines()[-1] if code.strip() else ""))
final = ex.get("final") or {}
print("  exit_code:", final.get("exit_code"))
print("  stdout 尾部:")
for ln in (final.get("stdout") or "").strip().splitlines()[-12:]:
    print("   ", ln)

print()
print("--- 3. ResultValidator ---")
va = data.get("validation", {}) or {}
print("  status:", va.get("status"))
print("  is_reproduced:", va.get("is_reproduced"))
print("  confidence:", va.get("confidence"))
mc = va.get("metrics_comparison", {}) or {}
print("  论文声明:", mc.get("paper"))
print("  实际运行:", mc.get("actual"))

print()
print("--- 4. 优化 ---")
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
print("--- 5. 报告（验证结论段）---")
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

os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write(report)
print()
print(f"完整报告已写入 {REPORT_OUT}")

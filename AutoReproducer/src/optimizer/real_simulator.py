"""RealSimulator - Optimizer 真实执行执行器（借鉴点 A3 + B1）。

替代 OptimizerAgent._simulate_trial 的哈希模拟：在复现工作区上真实执行
"LLM 补丁 -> 代码重跑 -> 提取真实指标 -> 计算改进奖励"，配合
src.safety 的补丁白名单与工作区快照回滚，构成可安全交付的真实优化闭环。

闭环（每次 trial）:
  1. snapshot_workspace 对工作区做 SHA-256 + 内容快照；
  2. LLM 生成针对 arm 优化方向的完整补丁代码（复用 CodeExecutor 清洗层）；
  3. PatchPolicy 白名单校验目标文件（受保护路径直接拒绝,不执行）；
  4. 写入目标文件并真实重跑（CodeExecutor.execute_in_workspace）；
  5. 从运行输出提取真实指标,计算相对改进率 reward(方向敏感:
     loss/mse/rmse 越小越好,其余越大越好);
  6. 无论成败 restore_snapshot 还原工作区,保证论文原始代码不被破坏;
     Keep(reward>0) 的补丁落盘到 workspace.parent/optimized_patches/ 供导出。

接口与 _simulate_trial 兼容：simulator(arm, baseline) -> (reward, detail)，
可直接注入 OptimizerAgent.simulator 而无需改动其 UCB 调度逻辑。
"""
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.agents.code_executor import CodeExecutorAgent
from src.agents.result_validator import ResultValidatorAgent
from src.safety.patch_policy import PatchPolicy
from src.safety.workspace_snapshot import restore_snapshot, snapshot_workspace

# 越小越好的指标键
_LOWER_IS_BETTER = {"loss", "mse", "rmse"}

_PATCH_PROMPT = """针对优化方向「{arm}」生成改进后的完整 Python 训练脚本。

当前基线脚本:
```python
{current_code}
```

【关键输出约束 - 必须严格遵守】
1. 输出一份**完整**的可直接运行脚本（完整训练+评估流程，末尾打印关键指标），
   不要为了简短而省略训练循环或评估步骤；
2. 用**单个** ```python 围栏把整份脚本包起来，围栏内只有代码；不要写围栏外的
   解释文字或中文叙述段落。中文字符**允许**出现在字符串字面量与 # 注释里；
3. 末尾必须以 `accuracy = 0.xxx` 或 `accuracy: 0.xxx` 形式打印改进后的指标。
"""


class RealSimulator:
    """在真实工作区执行优化补丁并返回改进奖励的执行器。"""

    def __init__(self, llm, executor: CodeExecutorAgent,
                 workspace_dir: str, target_file: str = "run.py",
                 policy: Optional[PatchPolicy] = None,
                 validator: Optional[ResultValidatorAgent] = None,
                 logger=None, patch_dir_name: str = "optimized_patches"):
        self.llm = llm
        self.executor = executor
        self.workspace = Path(workspace_dir)
        self.target_file = target_file
        # 白名单默认只放行目标文件;调用方可注入更严格的自定义策略
        self.policy = policy or PatchPolicy(editable=[target_file])
        self.validator = validator or ResultValidatorAgent(llm)
        self.logger = logger
        self.patch_dir = self.workspace.parent / patch_dir_name
        # 指标键(方向与对齐依据);None 时从运行输出自动推断
        self._metric_key: Optional[str] = None
        self._paper_info: Dict = {}

    # ---------------- 上下文绑定 ----------------

    def bind_paper(self, paper_info: Dict,
                   metric_key: Optional[str] = None) -> None:
        """绑定论文信息与奖励指标键(Orchestrator 进入优化阶段前调用)。"""
        self._paper_info = dict(paper_info or {})
        if metric_key is None:
            metrics = self._paper_info.get("metrics", {}) or {}
            if metrics:
                metric_key = str(next(iter(metrics)))
        self._metric_key = metric_key

    # ---------------- 主入口 ----------------

    def __call__(self, arm: str, baseline: float) -> Tuple[float, Dict]:
        """与 OptimizerAgent._run_trial 的 simulator 接口兼容。"""
        return self.run_trial(arm, baseline)

    def run_trial(self, arm: str, baseline: float) -> Tuple[float, Dict]:
        """执行一次真实优化尝试,返回 (reward, detail)。

        reward > 0 表示相对基线的真实改进(Keep),否则 Reject。
        """
        detail: Dict = {"type": "real_exec", "arm": arm,
                        "baseline": baseline}
        if not self.workspace.is_dir():
            detail.update({"status": "rejected",
                           "note": f"工作区不存在: {self.workspace}"})
            return 0.0, detail

        target = self.workspace / self.target_file
        if not target.exists():
            detail.update({"status": "rejected",
                           "note": f"缺少待优化文件: {self.target_file}"})
            return 0.0, detail

        original = target.read_text(encoding="utf-8")

        # 1) 安全网：工作区快照
        snap = snapshot_workspace(self.workspace)
        detail["snapshot_files"] = len(snap)

        # 2) LLM 生成补丁(完整新代码)
        patch = self.llm.chat(self._patch_prompt(arm, original),
                              task="optimizer_patch")
        code = self.executor._sanitize_code(patch)
        detail["patch_len"] = len(code)

        # 3) 补丁白名单 + 有效性检查
        ok, reason = self.policy.check(self.target_file)
        if not ok:
            detail.update({"status": "policy_rejected", "reason": reason})
            return 0.0, detail
        if not code or code == original:
            detail.update({"status": "rejected", "note": "补丁为空或与基线无变化"})
            return 0.0, detail

        # 3b) 语法门：不可编译的补丁**不写盘、不执行**。
        # 旧实现直接把补丁写到工作区再跑，运行期才炸 SyntaxError —— 一次
        # trial 白白烧掉，而且残码还落在了工作区里。（快照回滚虽能兜住，
        # 但没必要先把垃圾写进去。）
        syntax_err = self.executor._syntax_error(code)
        if syntax_err:
            detail.update({"status": "rejected",
                           "reason": f"补丁不可编译，未执行: {syntax_err}",
                           "patch_len": len(code)})
            if self.logger:
                self.logger.log("Optimizer", "patch_syntax_gate", "WARNING",
                                f"补丁未通过语法门，已拒绝: {syntax_err}")
            return 0.0, detail

        # 4) 应用补丁并真实重跑
        target.write_text(code, encoding="utf-8")
        stdout, success = "", False
        try:
            result = self.executor.execute_in_workspace(
                code, str(self.workspace), stage="full")
            stdout = result.get("stdout", "") or ""
            success = bool(result.get("success", False))
        except Exception as e:      # 执行器异常 -> 视作失败
            detail["exec_error"] = str(e)[:200]

        # 5) 真实指标与奖励
        reward, metric = self._compute_reward(baseline, stdout, success)
        detail["metric"] = metric

        # 6) 安全网：还原工作区(无论成败);Keep 的补丁落盘供导出
        restored, removed = restore_snapshot(self.workspace, snap)
        detail["restored"] = restored
        detail["removed"] = removed
        if reward > 0:
            detail["kept_patch"] = str(self._persist_patch(arm, code, metric))
        detail.update({
            "status": "kept" if reward > 0 else "rejected",
            "stdout_tail": stdout[-200:],
        })
        return reward, detail

    # ---------------- 内部工具 ----------------

    def _patch_prompt(self, arm: str, current_code: str) -> str:
        return _PATCH_PROMPT.format(arm=arm, current_code=current_code[:3000])

    def _compute_reward(self, baseline: float, stdout: str,
                        success: bool) -> Tuple[float, Optional[Dict]]:
        """从真实运行输出提取指标并计算相对改进率 reward。

        方向: loss/mse/rmse 越小越好;其余键越大越好。
        口径: 与 ResultValidator._local_compare 一致——一方小数(0~1)一方
        百分数(>=10)时归一到小数再对比。
        """
        if not success:
            return 0.0, None
        metrics = self.validator._extract_metrics(stdout)
        if not metrics:
            return 0.0, None

        key = self._metric_key if self._metric_key in metrics else None
        if key is None:     # 退而求其次:取与基线绝对值最接近的指标
            key = min(metrics, key=lambda k: abs(float(metrics[k]) - baseline))
        new = float(metrics[key])

        b, n = float(baseline), new
        if b <= 1.0 and n >= 10.0:
            n = n / 100.0
        elif b >= 10.0 and n <= 1.0:
            b = b / 100.0

        direction = -1.0 if key in _LOWER_IS_BETTER else 1.0
        denom = abs(b) if abs(b) > 1e-9 else 1e-9
        reward = round((n - b) / denom * direction, 4)
        return reward, {key: round(n, 6)}

    def _persist_patch(self, arm: str, code: str,
                       metric: Optional[Dict]) -> Path:
        """把 Keep 的补丁落盘到 optimized_patches/ 目录(工作区外)。"""
        safe_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", arm).strip("_")[:40] \
            or "patch"
        self.patch_dir.mkdir(parents=True, exist_ok=True)
        path = self.patch_dir / f"{safe_name}.py"
        path.write_text(code, encoding="utf-8")
        if self.logger:
            self.logger.log("Optimizer", "persist_patch", "SUCCESS",
                            f"Keep 补丁落盘: {path.name}", {"metric": metric})
        return path
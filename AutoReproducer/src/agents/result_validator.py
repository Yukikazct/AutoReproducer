"""ResultValidatorAgent - 结果验证 Agent，比对论文声明值与运行结果。

对齐方案「Phase 5: 结果验证」：
- 将实际运行结果与论文声明数值比对；
- 输出复现报告：成功/失败、数值差异、可能原因分析；
- 判断标准：环境是否 OK、输出是否 OK、描述是否 OK（输出 is_reproduced）。
"""
import json
import re
from typing import Dict
from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient
from src.metric_keys import norm_metric_key

# 指标提取模式：键名 -> 输出中的统一指标名
# 注意 rmse 必须排在 mse 之前，否则 "rmse: 1.2" 会被 mse 分支抢先匹配
_METRIC_PATTERNS = [
    (r"(?:accuracy|acc|精确率|准确率|测试集准确率)\s*[:：=]?\s*([\d.]+)\s*%?", "accuracy"),
    (r"(?:f1[_-]?score|f1)\s*[:：=]?\s*([\d.]+)", "f1_score"),
    (r"(?:precision|精确率)\s*[:：=]?\s*([\d.]+)\s*%?", "precision"),
    (r"(?:recall|召回率)\s*[:：=]?\s*([\d.]+)\s*%?", "recall"),
    (r"(?:loss|损失)\s*[:：=]?\s*([\d.]+)", "loss"),
    # \b 不可省：否则 "rmse: 1.2" 里的 "mse" 会被下面的 mse 分支抢先命中
    (r"\b(?:rmse|root[_\s]?mean[_\s]?squared[_\s]?error)\s*[:：=]?\s*([\d.]+)", "rmse"),
    (r"\b(?:mse|mean[_\s]?squared[_\s]?error)\s*[:：=]?\s*([\d.]+)", "mse"),
]
# 复现成功判定的相对差异阈值
_TOLERANCE = 0.05


class ResultValidatorAgent(BaseAgent):
    """验证运行结果是否与论文声明一致。"""

    system_prompt = "比对实际运行指标与论文声明值,输出复现成功/失败、数值差异与原因分析"

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("ResultValidator", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """验证执行结果。

        input_data: {"paper_info": dict, "execution": dict, "corpus_paper": str}
        """
        self.log("validate", "START", "开始验证结果", input_data)

        paper_info = input_data.get("paper_info", {}) or {}
        execution = input_data.get("execution", {}) or {}

        # execution 兼容新结构（stages）与旧结构（execution.execution）
        stdout = execution.get("stdout", "") or ""
        stderr = execution.get("stderr", "") or ""
        if not stdout and execution.get("stages"):
            stdout = execution["stages"][-1].get("stdout", "")
            stderr = execution["stages"][-1].get("stderr", "")

        paper_metrics = dict(paper_info.get("metrics", {}) or {})

        # 语料对照层：无声明指标时用语料复现分作为论文声明值
        corpus_paper = input_data.get("corpus_paper") or paper_info.get("corpus_paper")
        if not paper_metrics and corpus_paper:
            from src.corpus import get_declared_score
            score = get_declared_score(corpus_paper)
            if score is not None:
                paper_metrics = {"reproduction_score": round(float(score), 4)}

        # 代码根本没跑起来（语法错误/危险调用被前置拦截，或沙箱启动即失败）
        # -> "无法验证"，不能报成"复现失败"——后者会误导用户以为方法不对。
        not_runnable = self._detect_not_runnable(execution, stdout)
        if not_runnable:
            reason = f"代码未能运行，无法与论文声明比对（原因：{not_runnable}）"
            self.log_experiment(
                "VALIDATE", "代码未运行,跳过指标比对",
                inputs={"execution_stage": (execution.get("final") or {}).get("stage")},
                outputs={"status": "not_runnable"},
                result={"is_reproduced": None, "reason": not_runnable})
            self.log("validate", "WARNING", reason)
            return {
                "validation": {"match": None, "differences": [],
                               "confidence": 0.0, "analysis": reason},
                "metrics_comparison": {"paper": paper_metrics, "actual": {}},
                "is_reproduced": None,
                "status": "not_runnable",
                "reason": not_runnable,
                "confidence": 0.0,
                "llm_calls": self._delta_llm_calls(),
            }

        # 信息不足下的"尽力而为"执行：代码确实跑了，但它是占位实现，不能拿去
        # 与论文声明比对 -> 第四态 best_effort（既不判成功也不判失败）。
        # 顺序即优先级：真没跑（not_runnable）> 跑了但不可核对（best_effort）>
        # 正常比对——"没跑起来"永远比"跑了个占位"更该优先告知用户。
        #
        # 为什么要这一态：不拦的话，占位脚本一旦打印出任何可提取的数值，
        # `_local_compare` 对"无论文声明指标"是乐观判定（跑出数值即 match=True）
        # -> 报告显示假的 ✅ 复现成功、还会真去触发优化；一个数值都抽不到时又
        # 显示假的 ❌ 失败。两种都是把"信息不足"翻译成了错误结论。
        if execution.get("best_effort") or paper_info.get("insufficient_info"):
            actual_metrics = self._extract_metrics(stdout)   # 证据照留
            reason = ("论文信息不足，代码为尽力而为的占位实现，其输出不能与"
                      "论文声明比对（不判定为复现成功或失败）")
            self.log_experiment(
                "VALIDATE", "信息不足,跳过结论判定",
                inputs={"stdout_tail": stdout[-300:]},
                outputs={"status": "best_effort",
                         "actual_metrics": actual_metrics},
                result={"is_reproduced": None, "reason": reason})
            self.log("validate", "WARNING", reason)
            return {
                # differences 是"逐项数值差异"，占位实现没有可比对的项；
                # 理由已经在 analysis/reason 里，塞进 differences 只会让报告
                # 把同一句话印三遍。
                "validation": {"match": None, "differences": [],
                               "confidence": 0.0, "analysis": reason},
                "metrics_comparison": {"paper": paper_metrics,
                                       "actual": actual_metrics},
                "is_reproduced": None,
                "status": "best_effort",
                "reason": reason,
                "confidence": 0.0,
                "llm_calls": self._delta_llm_calls(),
            }

        actual_metrics = self._extract_metrics(stdout)

        # LLM 比对 + 本地数值校验兜底。
        # 判据必须写进 prompt：不写的话模型只能凭感觉判，实测同样的输入
        # （声明 0.0892 / 实际 0.0869）会在 true/false 之间反复横跳，而它
        # 与本地规则取交集，一次 false 就把正确结论否决掉。
        prompt = f"""比对论文声明的指标与代码运行结果。

论文声明指标: {json.dumps(paper_metrics, ensure_ascii=False)}
代码运行输出: {stdout[:2000]}
提取到的实际指标: {json.dumps(actual_metrics, ensure_ascii=False)}

判定规则（必须严格遵守，不要自行加严或放宽）:
1. 以"提取到的实际指标"为运行结果的准据。输出里若同时出现论文声明值
   （例如复现脚本自己打印了一行声明指标做对照），**不得**把它当成运行结果。
2. 指标名的大小写与分隔符差异不构成不同指标：MSE 与 mse、F1_score 与
   "F1 Score" 是同一个指标，必须照常比对。
3. 逐项算相对差异 |实际-声明|/|声明|：
   - ≤ {_TOLERANCE:.0%} 视为一致（实验存在随机性，这是正常波动）；
   - > {_TOLERANCE:.0%} 视为不一致；
   - 声明了但确实没跑出该指标，视为不一致（无法证实），并在 differences 里写明。
4. match=true 当且仅当所有声明指标都一致。只要有一项超阈值或无对应输出，
   match 必须为 false。
5. 分析里给出每个指标的实际相对差异百分比，不要只说"接近"或"有差异"。

返回JSON格式:
{{
    "match": true/false,
    "differences": ["指标1: 声明值 vs 实际值 (相对差异 X%)"],
    "confidence": 0.0-1.0,
    "analysis": "分析说明"
}}
"""
        llm_result = self.llm.chat(prompt, task="result_validator")
        parsed = self._parse_json(llm_result)
        if not parsed or "match" not in parsed:
            parsed = self._local_compare(paper_metrics, actual_metrics)

        # 本地校验：与 LLM 结果取交集（两者都判成功才算成功）
        local = self._local_compare(paper_metrics, actual_metrics)
        llm_ok = bool(parsed.get("match", False))
        local_ok = bool(local.get("match", False))
        match = llm_ok and local_ok
        if paper_metrics and not actual_metrics:
            match = False  # 有声明无实测值 -> 不可判定为复现成功

        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        differences = list(parsed.get("differences", []) or [])
        # 两个独立判据结论相反时，如实记下分歧并按"最弱一环"报置信度。
        # 否则会出现"LLM 说 match=true/置信度 1.0，最终却报未复现且置信度 1.0"
        # 这种自相矛盾的结论——用户无法分辨到底是"确定没复现"还是"判据打架"。
        if llm_ok != local_ok:
            differences.append(
                f"判据分歧: 模型判定 match={llm_ok}，本地数值比对判定 "
                f"match={local_ok}；以交集为准（{match}）")
            confidence = min(confidence, float(local.get("confidence", 0.0) or 0.0))

        result = {
            "validation": {**parsed, "match": match, "differences": differences,
                           "verdict_sources": {"llm": llm_ok, "local": local_ok}},
            "metrics_comparison": {"paper": paper_metrics, "actual": actual_metrics},
            "is_reproduced": match,
            "status": "reproduced" if match else "not_reproduced",
            "confidence": round(confidence, 4),
        }

        self.log_experiment(
            "VALIDATE", "比对论文声明与运行结果",
            inputs={"paper_metrics": paper_metrics, "stdout_tail": stdout[-500:]},
            outputs=actual_metrics,
            result={"is_reproduced": match, "differences": parsed.get("differences", [])},
        )
        self.log("validate",
                 "SUCCESS" if match else "WARNING",
                 f"验证{'通过' if match else '未通过'} - 置信度: {result['confidence']:.2f}",
                 result)

        return {**result, "llm_calls": self._delta_llm_calls()}

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _detect_not_runnable(execution: Dict, stdout: str) -> str:
        """判断执行是否"压根没跑起来"；是则返回原因文本，否则返回 ""。

        与"跑起来了但结果不符"区分：只有前者才应报"无法验证"。判据：
        1. CodeExecutor 前置门拦下（not_runnable 标记 / exit_code=-5）；
        2. 最终阶段失败且没有任何 stdout（依赖装不上、语法错误、超时等）。

        注意"跑了但信息不足"（best_effort）**不**属于这里：那是另一态，由
        `run()` 里的 best_effort 分支处理，本方法不该把它报成"没跑起来"。
        """
        if execution.get("not_runnable"):
            return (execution.get("reason") or "代码未进入执行阶段").strip()
        final = execution.get("final") or {}
        if final.get("not_runnable"):
            return (final.get("stderr") or "代码未进入执行阶段").strip()
        if final and not final.get("success") and not (stdout or "").strip():
            detail = (final.get("stderr") or "").strip()
            return (detail[:200] if detail else "执行未产出任何输出")
        return ""

    @staticmethod
    def _norm_metric_key(key) -> str:
        """指标键归一（规则见 `src/metric_keys.py`，与报告展示层同源）。

        实测（真实模式）：论文声明 `{"MSE": 0.0892}`，运行输出打印
        `MSE: 0.0869`，经 `_extract_metrics` 归一成键 `mse`；而这里原先用
        `akey == key` 精确比对，`"MSE" != "mse"` 于是判"声明了但没提取到"，
        把 2.6%（远小于 5% 阈值）的差异**误报成未复现**，理由还写反了。
        """
        return norm_metric_key(key)

    def _local_compare(self, paper_metrics: Dict, actual_metrics: Dict) -> Dict:
        """本地规则比对：同键指标相对差异 <= 5% 视为匹配。"""
        if not paper_metrics:
            return {"match": bool(actual_metrics), "differences": [],
                    "confidence": 0.8 if actual_metrics else 0.3,
                    "analysis": "无论文声明指标,以运行产出是否有效判定"}
        if not actual_metrics:
            return {"match": False, "differences": ["论文声明指标但运行输出未提取到数值"],
                    "confidence": 0.3, "analysis": "运行输出缺少可解析的数值指标"}

        # 键归一后再比对（大小写/分隔符不敏感）；同归一键取首次出现值
        norm_actual: Dict[str, object] = {}
        for akey, aval in actual_metrics.items():
            norm_actual.setdefault(self._norm_metric_key(akey), aval)

        differences, missing = [], []
        match_all = True
        matched = 0
        for key, declared in paper_metrics.items():
            try:
                declared_num = float(declared)
            except (TypeError, ValueError):
                continue
            actual = None
            aval = norm_actual.get(self._norm_metric_key(key))
            if aval is not None:
                try:
                    actual = float(aval)
                except (TypeError, ValueError):
                    actual = None
            if actual is None:
                # 声明了但输出里没提取到：如实记为"无法比对"（不再静默跳过）
                missing.append(
                    f"{key}: 论文声明 {declared_num:g}，运行输出未提取到该指标")
                continue
            matched += 1
            # 口径统一：一方为小数(0~1)、另一方为百分数(>=10)时,归一到小数再比对
            declared_raw, actual_raw = declared_num, actual
            if declared_num <= 1.0 and actual >= 10.0:
                actual = actual / 100.0
            elif declared_num >= 10.0 and actual <= 1.0:
                declared_num = declared_num / 100.0
            diff = abs(actual - declared_num) / max(abs(declared_num), 1e-9)
            if diff > _TOLERANCE:
                match_all = False
            differences.append(
                f"{key}: 声明 {declared_raw:g} vs 实际 {actual_raw:g} "
                f"(归一化后相对差异 {diff:.1%})")

        if matched == 0:
            # 声明指标一个都没对上 -> 不是"复现成功",而是数据对不上号
            match_all = False
        differences += missing
        return {"match": match_all, "differences": differences,
                "missing_metrics": missing,
                "confidence": 0.8 if match_all else 0.4,
                "analysis": ("本地数值比对完成"
                             + (f"；{len(missing)} 个声明指标未提取到" if missing
                                else ""))}

    def _extract_metrics(self, text: str) -> Dict:
        """从输出文本中提取指标数值。"""
        metrics: Dict[str, float] = {}
        for pattern, name in _METRIC_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics[name] = float(match.group(1))
                except ValueError:
                    pass
        # 覆盖键=值 风格的行（如 accuracy=0.852）
        for line in (text or "").splitlines():
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\d.]+)\s*$", line)
            if m:
                try:
                    metrics[m.group(1).lower()] = float(m.group(2))
                except ValueError:
                    pass
        # 值为 0~1 的 accuracy 统一保留（用于与声明 0.85 对齐）
        return metrics

    def _delta_llm_calls(self) -> int:
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    @staticmethod
    def _parse_json(text: str) -> Dict:
        for candidate in (text, re.sub(r"```(?:json)?\s*(.*?)```", r"\1", text, flags=re.DOTALL)):
            if not candidate or not candidate.strip():
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                match = re.search(r"\{.*\}", candidate, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
        return {}
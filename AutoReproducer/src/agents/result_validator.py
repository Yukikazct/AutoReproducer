"""结果验证：先核验执行证据，再按显式单位确定性比较最终指标。"""
import json
import re
from typing import Dict

from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient
from src.metric_keys import norm_metric_key
from src.runtime_metrics import (
    comparable_values, metric_number, metric_unit, parse_runtime_metrics,
)

_TOLERANCE = 0.05


class ResultValidatorAgent(BaseAgent):
    """模型解释偏差；执行状态、指标完整性和数值规则决定结论。"""

    system_prompt = "核验执行证据和最终指标，确定性比对论文数值，模型只解释偏差"

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("ResultValidator", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        self.log("validate", "START", "开始验证结果", input_data)
        paper_info = input_data.get("paper_info", {}) or {}
        execution = input_data.get("execution", {}) or {}
        final = self._final_execution(execution)
        stdout = final.get("stdout", "") or ""
        paper_metrics = dict(paper_info.get("metrics", {}) or {})
        paper_units = paper_info.get("metric_units", {}) or {}
        structured = final.get("metric_records", final.get("metrics"))
        records, metric_errors = parse_runtime_metrics(stdout, structured, expected_names=paper_metrics)
        actual_metrics = {r["name"]: r["value"] for r in records if r["value"] is not None}
        actual_units = {r["name"]: r["unit"] for r in records}
        execution_status, execution_reason = self._execution_state(execution)

        def finish(status, match, reason, comparison=None, level=None):
            inner = {**(comparison or {}), "match": match, "analysis": reason,
                     "verdict_source": "deterministic", "relative_tolerance": _TOLERANCE}
            inner.setdefault("differences", [])
            inner.setdefault("confidence", 0.0)
            result = {
                "validation": inner,
                "metrics_comparison": {"paper": paper_metrics, "actual": actual_metrics,
                                       "paper_units": paper_units, "actual_units": actual_units},
                "metric_records": records,
                "is_reproduced": match, "status": status, "reason": reason,
                "execution_status": execution_status,
                "execution_source": execution.get("execution_source"),
                "reproduction_scope": execution.get("reproduction_scope"),
                "repository_executed": execution.get("repository_executed", False),
                "result_level": level or ("reproduced" if match else "inconclusive"),
                "confidence": inner["confidence"], "llm_calls": self._delta_llm_calls(),
            }
            self.log_experiment("VALIDATE", reason, inputs={"paper_metrics": paper_metrics},
                                outputs={"actual_metrics": actual_metrics,
                                         "metric_records": records},
                                result={"status": status, "is_reproduced": match})
            self.log("validate", "SUCCESS" if match is True else "WARNING", reason, result)
            return result

        if execution.get("evidence_status") == "insufficient_evidence":
            execution_status = "not_run"
            return finish("insufficient_evidence", None,
                          execution.get("reason") or "论文证据不足，未获得可运行实现；未执行论文实验。")
        not_runnable = self._detect_not_runnable(execution, stdout)
        if not_runnable:
            return finish("not_runnable", None,
                          f"代码未能运行，无法与论文声明比对（原因：{not_runnable}）",
                          level="failed")
        if execution_status == "failed":
            return finish("execution_failed", False,
                          f"执行失败，不能判定论文数值复现：{execution_reason}", level="failed")
        if execution_status == "incomplete":
            return finish("execution_incomplete", None,
                          f"执行证据不足，无法核验：{execution_reason}")
        if execution.get("best_effort") or execution.get("fallback_used") or paper_info.get("insufficient_info"):
            return finish("best_effort", None,
                          "论文信息不足或代码使用了演示兜底，其输出不能与论文声明比对"
                          "（不判定为复现成功或失败）")
        if metric_errors:
            return finish("invalid_metrics", False, "最终运行指标无效：" + "；".join(metric_errors),
                          {"differences": metric_errors, "invalid_metrics": metric_errors},
                          level="failed")
        if execution_status == "smoke_passed":
            return finish("smoke_passed", None,
                          "仅完成冒烟检查，尚未完成完整实验与论文数值核验。", level="smoke_passed")

        required = paper_info.get("required_metrics", []) or []
        local = self._local_compare(paper_metrics, actual_metrics, paper_units, actual_units,
                                    required_metrics=required)
        norm_records = {r["name"]: r for r in records}
        for key, declared in paper_metrics.items():
            record = norm_records.get(norm_metric_key(key))
            if not record:
                continue
            expected_split = declared.get("split") if isinstance(declared, dict) else None
            expected_stage = declared.get("stage") if isinstance(declared, dict) else None
            errors = []
            if record["stage"] == "train":
                errors.append(f"{key}: 只有训练指标，缺少最终评估指标")
            if expected_split and record["split"] != str(expected_split).lower():
                errors.append(f"{key}: 数据划分不一致（要求 {expected_split}，实际 {record['split'] or '未标注'}）")
            if expected_stage and record["stage"] != str(expected_stage).lower():
                errors.append(f"{key}: 指标阶段不一致（要求 {expected_stage}，实际 {record['stage'] or '未标注'}）")
            if errors:
                local["match"] = False
                local["confidence"] = 0.4
                local.setdefault("invalid_metrics", []).extend(errors)
                local["differences"].extend(errors)
                local["analysis"] = "指标来源不满足最终评估要求"

        if not paper_metrics and local["match"] is not False:
            # 语料库复现评分是评测项目得分，不能充当上传论文的性能参考数值。
            return finish("no_reference_metrics", None,
                          "论文未声明参考指标数值，无法核验运行结果是否与论文一致。"
                          "已保留实际运行指标；需补充参考数值后才能判断复现结论。")
        if local.get("missing_metrics") or local.get("invalid_metrics"):
            return finish("not_reproduced", False, local["analysis"], local, level="failed")

        # 模型输出不能改变已经确定的 match，也不能替换逐项数值差异。
        prompt = f"""比对论文声明的指标与代码运行结果，解释以下确定性数值判定。
论文声明指标: {json.dumps(paper_metrics, ensure_ascii=False)}
论文指标单位: {json.dumps(paper_units, ensure_ascii=False)}
提取到的实际指标: {json.dumps(actual_metrics, ensure_ascii=False)}
实际指标单位: {json.dumps(actual_units, ensure_ascii=False)}
确定性结果: {json.dumps(local, ensure_ascii=False)}

规则：所有声明指标都一致、执行成功且所有必需指标有限时才可通过。
相对差异 |实际-声明|/|声明| ≤ {_TOLERANCE:.0%} 视为一致。
指标名的大小写与分隔符差异不构成不同指标；仅显式百分比/比例单位允许换算。
输出中的论文声明值不得当作运行结果。仅冒烟通过不等于论文数值复现。
你只解释偏差，不得覆盖确定性判定。返回 JSON：
{{"match": true/false, "analysis": "原因解释"}}
"""
        parsed = {}
        try:
            parsed = self._parse_json(self.llm.chat(prompt, task="result_validator"))
        except Exception as exc:
            self.log("explain_metrics", "WARNING", f"模型解释不可用，保留确定性判定：{type(exc).__name__}")
        llm_match = parsed.get("match") if isinstance(parsed.get("match"), bool) else None
        local["verdict_sources"] = {"llm": llm_match, "local": local["match"]}
        if llm_match is not None and llm_match != local["match"]:
            local["differences"].append(
                f"判据分歧: 模型解释 match={llm_match}，确定性数值判定 match={local['match']}；采用确定性判定")
        elif llm_match is local["match"] and isinstance(parsed.get("analysis"), str) and parsed["analysis"].strip():
            local["analysis"] += "；" + parsed["analysis"].strip()
        match = local["match"]
        return finish("reproduced" if match else "not_reproduced", match, local["analysis"], local,
                      level="reproduced" if match else "experiment_completed")

    @staticmethod
    def _final_execution(execution):
        if execution.get("final"):
            return execution["final"]
        if execution.get("stages"):
            return execution["stages"][-1]
        return execution.get("execution") or execution

    @classmethod
    def _execution_state(cls, execution):
        final = cls._final_execution(execution)
        history = execution.get("steps") or execution.get("stages") or []
        # 自愈历史中的旧失败已经由新一轮重跑替代，不能污染最终结论。
        if history and final.get("repair_round") is not None:
            history = [st for st in history if st.get("repair_round") == final["repair_round"]]
        current = {}
        for index, step in enumerate(history):
            current[step.get("id") or step.get("stage") or index] = step
        checks = [final, *current.values(), execution]
        for step in checks:
            if step.get("required") is False:
                continue
            code = step.get("exit_code")
            failed = step.get("success") is False or step.get("timed_out") or step.get("cancelled")
            if code is not None and code != 0:
                failed = True
            if failed:
                name = step.get("id") or step.get("stage") or "执行"
                detail = str(step.get("stderr") or step.get("reason") or "").strip()[:200]
                return "failed", f"{name} 失败（退出码 {code if code is not None else '未知'}）" + (f"：{detail}" if detail else "")
        required = [st for st in current.values() if st.get("required") is not False]
        if not final or not (final.get("exit_code") == 0 or final.get("success") is True):
            return "incomplete", "缺少最终阶段成功状态或退出码"
        if any(not (st.get("exit_code") == 0 or st.get("success") is True) for st in required):
            return "incomplete", "存在未完成的必需步骤"
        stage = str(final.get("stage") or "").lower()
        if stage in {"smoke", "import_check", "check", "precheck"} or execution.get("validation_level") == "smoke":
            return "smoke_passed", ""
        if stage in {"train", "training", "prepare", "dependencies"}:
            return "incomplete", "仅完成训练或准备步骤，缺少最终评估执行证据"
        return "experiment_completed", ""

    @classmethod
    def _detect_not_runnable(cls, execution: Dict, stdout: str) -> str:
        if execution.get("not_runnable"):
            return (execution.get("reason") or "代码未进入执行阶段").strip()
        final = cls._final_execution(execution)
        if final.get("not_runnable"):
            return (final.get("stderr") or "代码未进入执行阶段").strip()
        if (final.get("success") is False or final.get("exit_code", 0) != 0) and not str(stdout).strip():
            return str(final.get("stderr") or "执行未产出任何输出").strip()[:200]
        return ""

    @staticmethod
    def _norm_metric_key(key) -> str:
        return norm_metric_key(key)

    def _local_compare(self, paper_metrics: Dict, actual_metrics: Dict,
                       paper_units=None, actual_units=None, required_metrics=None) -> Dict:
        norm_actual = {norm_metric_key(k): v for k, v in actual_metrics.items()}
        p_units = {norm_metric_key(k): v for k, v in (paper_units or {}).items()}
        a_units = {norm_metric_key(k): v for k, v in (actual_units or {}).items()}
        differences, missing, invalid = [], [], []
        match = bool(paper_metrics)
        for key in required_metrics or []:
            if norm_metric_key(key) not in norm_actual:
                missing.append(f"{key}: 运行输出未提取到必需指标")
        for key, declared in paper_metrics.items():
            normalized = norm_metric_key(key)
            declared_num = metric_number(declared)
            if declared_num is None:
                invalid.append(f"{key}: 论文参考值不是有限数值")
                continue
            if normalized not in norm_actual:
                missing.append(f"{key}: 论文声明 {declared_num:g}，运行输出未提取到该指标")
                continue
            actual = norm_actual[normalized]
            actual_num = metric_number(actual)
            if actual_num is None:
                invalid.append(f"{key}: 实际指标不是有限数值")
                continue
            p_unit = metric_unit(declared, p_units.get(normalized))
            a_unit = metric_unit(actual, a_units.get(normalized))
            reference, measured, error = comparable_values(declared_num, actual_num, p_unit, a_unit)
            if error:
                invalid.append(f"{key}: {error}")
                continue
            diff = abs(measured - reference) / abs(reference) if reference else (0.0 if measured == 0 else float("inf"))
            if diff > _TOLERANCE:
                match = False
            differences.append(f"{key}: 声明 {declared_num:g} vs 实际 {actual_num:g} (相对差异 {diff:.1%})")
        if missing or invalid:
            match = False
        differences += missing + invalid
        if not paper_metrics and not missing and not invalid:
            match = None
        analysis = "本地数值比对完成"
        if missing:
            analysis += f"；{len(missing)} 个必需指标未提取到"
        if invalid:
            analysis += f"；{len(invalid)} 个指标值或单位无效"
        return {"match": match, "differences": differences, "missing_metrics": missing,
                "invalid_metrics": invalid, "confidence": 0.8 if match else (0.4 if match is False else 0.0),
                "analysis": analysis}

    def _extract_metrics(self, text: str) -> Dict:
        records, _ = parse_runtime_metrics(text)
        return {r["name"]: r["value"] for r in records if r["value"] is not None}

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

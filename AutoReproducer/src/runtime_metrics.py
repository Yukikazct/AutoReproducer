"""最终运行指标：结构化输出优先，文本回退保留单位和来源。"""
import json
import math
import re

from src.metric_keys import norm_metric_key

METRICS_MARKER = "AUTOREPRO_METRICS:"
NUMBER_PATTERN = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
_METRIC_RE = re.compile(
    r"(?<![\w])(?P<name>root[_\s-]?mean[_\s-]?squared[_\s-]?error|"
    r"mean[_\s-]?squared[_\s-]?error|测试集准确率|准确率|精确率|召回率|损失|"
    r"[A-Za-z_][A-Za-z0-9_-]*(?:\s+score)?)"
    r"\s*(?P<sep>[:：=])\s*(?P<value>" + NUMBER_PATTERN + r")(?![\w.])\s*(?P<percent>%)?",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(r"论文|声明|参考|目标|\b(?:paper|declared|reference|target)\b", re.I)
_FINAL_STAGES = {"final", "full", "test", "eval", "evaluation"}
_TEXT_METRICS = {"accuracy", "f1_score", "precision", "recall", "loss", "mse", "rmse", "mae",
                 "r2", "auc", "roc_auc", "mape", "smape", "psnr", "ssim", "bleu", "rouge",
                 "perplexity", "score", "reproduction_score"}


def metric_number(value):
    """布尔、无效字符串、NaN/Inf 都不属于可比较的数值。"""
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().removesuffix("%").strip()
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def metric_unit(value=None, explicit=None):
    if explicit is None and isinstance(value, dict):
        explicit = value.get("unit")
        value = value.get("value")
    if explicit is None and isinstance(value, str) and value.strip().endswith("%"):
        explicit = "percent"
    unit = str(explicit or "").strip().lower()
    return {"%": "percent", "percentage": "percent", "百分比": "percent",
            "ratio": "fraction", "0-1": "fraction", "比例": "fraction",
            "raw": "", "unspecified": ""}.get(unit, unit)


def comparable_values(declared, actual, paper_unit="", actual_unit=""):
    """只有显式 percent/fraction 单位可以转换；绝不按数值大小猜单位。"""
    paper_unit, actual_unit = metric_unit(explicit=paper_unit), metric_unit(explicit=actual_unit)
    if paper_unit == actual_unit:
        return declared, actual, ""
    if {paper_unit, actual_unit} == {"percent", "fraction"}:
        return (declared / 100 if paper_unit == "percent" else declared,
                actual / 100 if actual_unit == "percent" else actual, "")
    return declared, actual, (
        f"单位不一致（论文：{paper_unit or '未标注'}，实际：{actual_unit or '未标注'}），"
        "需显式定义单位后才能换算")


def _select_records(records):
    selected = {}
    ranks = {}
    for record in records:
        key = norm_metric_key(record["name"])
        # 最终评估优先；同一阶段的多轮输出取最后一个值。
        stage = str(record.get("stage") or "").lower()
        split = str(record.get("split") or "").lower()
        rank = (stage in _FINAL_STAGES, split in {"test", "eval", "evaluation"})
        if key not in ranks or rank >= ranks[key]:
            selected[key], ranks[key] = {**record, "name": key}, rank
    return list(selected.values())


def _structured_records(payload, source):
    errors, records = [], []
    if isinstance(payload, dict) and "metrics" in payload:
        if payload.get("version", 1) != 1:
            return [], ["不支持的结构化指标协议版本"]
        payload = payload["metrics"]
    if isinstance(payload, dict):
        entries = [dict(value, name=name) if isinstance(value, dict)
                   else {"name": name, "value": value} for name, value in payload.items()]
    elif isinstance(payload, list):
        entries = payload
    else:
        return [], ["结构化指标必须是指标列表或名称到数值的映射"]
    for entry in entries:
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            errors.append("结构化指标缺少 name")
            continue
        value = metric_number(entry.get("value"))
        if value is None:
            errors.append(f"{entry['name']}: 指标不是有限数值")
        record = {**entry, "value": value, "unit": metric_unit(entry), "source": source,
                  "stage": str(entry.get("stage") or "final").lower(),
                  "split": str(entry.get("split") or "").lower()}
        records.append(record)
    records = _select_records(records)
    for record in records:
        if record["stage"] not in _FINAL_STAGES:
            errors.append(f"{record['name']}: 结构化指标来自 {record['stage']} 阶段，缺少最终评估指标")
    return records, errors


def parse_runtime_metrics(text, structured=None, expected_names=()):
    """返回 (最终指标记录, 解析错误)。损坏的结构化输出不能回退到历史文本。"""
    if structured is not None:
        return _structured_records(structured, "execution.metrics")
    lines = str(text or "").splitlines()
    marked = [(n, line.strip()[len(METRICS_MARKER):].strip())
              for n, line in enumerate(lines, 1) if line.strip().startswith(METRICS_MARKER)]
    if marked:
        line_number, payload = marked[-1]
        try:
            decoded = json.loads(payload)
        except (ValueError, TypeError):
            return [], [f"stdout:{line_number}: 最终结构化指标不是合法 JSON"]
        return _structured_records(decoded, f"stdout:{line_number}")

    records = []
    known_names = _TEXT_METRICS | {norm_metric_key(key) for key in expected_names}
    for line_number, line in enumerate(lines, 1):
        for match in _METRIC_RE.finditer(line):
            # 脚本自印的论文参考值不能覆盖真实测量。
            prefix = re.split(r"[,;，；]", line[:match.start()])[-1]
            if _REFERENCE_RE.search(prefix):
                continue
            name = norm_metric_key(match["name"])
            if name in {"epoch", "step", "seed", "version"}:
                continue
            if name not in known_names and match["sep"] != "=":
                continue
            stage = "final" if re.search(r"\bfinal\b|最终", prefix, re.I) else ""
            split = ""
            if re.search(r"\b(?:test|eval|evaluation)\b|测试|评估", prefix, re.I):
                split, stage = "test", "final"
            elif re.search(r"\b(?:train|training)\b|训练", prefix, re.I):
                split, stage = "train", "train"
            records.append({"name": name, "value": metric_number(match["value"]),
                            "unit": "percent" if match["percent"] else "",
                            "split": split, "stage": stage, "source": f"stdout:{line_number}"})
    records = _select_records(records)
    errors = [f"{r['name']}: 指标不是有限数值" for r in records if r["value"] is None]
    return records, errors

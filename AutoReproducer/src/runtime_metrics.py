"""最终运行指标：结构化输出优先，文本回退保留单位和来源。"""
import json
import math
import re

from src.metric_keys import norm_metric_key

METRICS_MARKER = "AUTOREPRO_METRICS:"
NUMBER_PATTERN = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
_METRIC_LABEL = (
    r"root[_\s-]?mean[_\s-]?squared[_\s-]?error|"
    r"mean[_\s-]?squared[_\s-]?error|测试集准确率|准确率|精确率|召回率|损失|"
    r"[A-Za-z_][A-Za-z0-9_-]*(?:\s+score)?")
_METRIC_RE = re.compile(
    r"(?<![\w])(?P<name>" + _METRIC_LABEL + r")"
    r"\s*(?P<sep>[:：=])\s*(?P<value>" + NUMBER_PATTERN + r")(?![\w.])\s*(?P<percent>%)?",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"(?<![\w])(?P<name>" + _METRIC_LABEL + r")"
    r"(?:\s+(?:of|over|across|from)\s+(?P<runs>\d+)\s+(?:runs|trials|seeds|repeats))?"
    r"\s*[:=]?\s+(?P<aggregation>mean|average|avg|median|best)\s*[:=]?\s*"
    r"(?P<value>" + NUMBER_PATTERN + r")(?![\w.])\s*(?P<percent>%)?", re.I)
_AGGREGATIONS = {'single', 'mean', 'median', 'best', 'unknown'}
_STAT_NAMES = {'mean', 'average', 'avg', 'std', 'stdev', 'stddev', 'median', 'best',
               'runs', 'trials', 'seeds', 'n'}
_RUNS_RE = re.compile(r"\b(?:(\d+)\s+(?:runs|trials|seeds|repeats)|"
                      r"(?:runs|trials|seeds|repeats|n)\s*[:=]\s*(\d+))\b", re.I)
_STD_RE = re.compile(r"(?:\b(?:std(?:dev)?|stdev|standard\s+deviation)\s*[:=]?\s*|"
                     r"(?:±|\+/-)\s*)(?P<std>" + NUMBER_PATTERN + r")(?![\w.])", re.I)
_OTHER_UNCERTAINTY = re.compile(r"\b(?:ci|confidence\s+interval|sem|stderr|standard\s+error)\b", re.I)
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


def _select_records(records, preserve_contexts=False):
    selected = {}
    ranks = {}
    for record in records:
        key = norm_metric_key(record["name"])
        # 最终评估优先；同一阶段的多轮输出取最后一个值。
        stage = str(record.get("stage") or "").lower()
        split = str(record.get("split") or "").lower()
        rank = (stage in _FINAL_STAGES, split in {"test", "eval", "evaluation"},
                record.get('aggregation') in {'mean', 'median', 'best'})
        identity = (key, split, record.get("dataset", ""), record.get("method", "")) if preserve_contexts else key
        if identity not in ranks or rank >= ranks[identity]:
            selected[identity], ranks[identity] = {**record, "name": key}, rank
    return list(selected.values())


def _statistics(prefix, suffix, summary=None):
    """Only explicit output supplies statistics; a process count proves no runs."""
    fields = {}
    aggregation = summary.group('aggregation') if summary is not None else None
    if aggregation is None:
        match = re.search(r'\b(mean|average|avg|median|best)\b', prefix, re.I)
        aggregation = match.group(1) if match else None
    if aggregation:
        fields['aggregation'] = {'average': 'mean', 'avg': 'mean'}.get(aggregation.lower(), aggregation.lower())
    counts = {int(a or b) for a, b in _RUNS_RE.findall(prefix + ' ' + suffix)}
    if summary is not None and summary.group('runs') is not None:
        counts.add(int(summary.group('runs')))
    if len(counts) == 1:
        fields['runs'] = counts.pop()
    std = _STD_RE.search(suffix)
    if std and not _OTHER_UNCERTAINTY.search(suffix):
        fields['std'] = metric_number(std.group('std'))
        # Unqualified mean ± standard deviation is a conventional summary,
        # while an explicitly labelled confidence interval remains unknown.
        fields.setdefault('aggregation', 'mean')
        if (std.group().lstrip().startswith(('±', '+/-'))
                and re.match(r'\s*%', suffix[std.end():])):
            fields['unit'] = 'percent'
    return fields


def _statistic_errors(record):
    errors = []
    if 'aggregation' in record and (not isinstance(record['aggregation'], str)
                                    or record['aggregation'] not in _AGGREGATIONS):
        errors.append('统计方式无效')
        record['aggregation'] = 'unknown'
    if 'runs' in record and (type(record['runs']) is not int or record['runs'] < 1):
        errors.append('重复次数必须是正整数')
        record['runs'] = None
    if 'std' in record:
        std = metric_number(record['std'])
        if std is None or std < 0:
            errors.append('标准差必须是有限非负数')
            std = None
        record['std'] = std
    return [f"{record['name']}: {message}" for message in errors]


def _structured_records(payload, source, preserve_contexts=False):
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
        errors.extend(_statistic_errors(record))
        records.append(record)
    records = _select_records(records, preserve_contexts)
    for record in records:
        if record["stage"] not in _FINAL_STAGES:
            errors.append(f"{record['name']}: 结构化指标来自 {record['stage']} 阶段，缺少最终评估指标")
    return records, errors


def parse_runtime_metrics(text, structured=None, expected_names=(), *, preserve_contexts=False):
    """返回 (最终指标记录, 解析错误)。损坏的结构化输出不能回退到历史文本。"""
    if structured is not None:
        return _structured_records(structured, "execution.metrics", preserve_contexts)
    lines = str(text or "").splitlines()
    marked = [(n, line.strip()[len(METRICS_MARKER):].strip())
              for n, line in enumerate(lines, 1) if line.strip().startswith(METRICS_MARKER)]
    if marked:
        line_number, payload = marked[-1]
        try:
            decoded = json.loads(payload)
        except (ValueError, TypeError):
            return [], [f"stdout:{line_number}: 最终结构化指标不是合法 JSON"]
        return _structured_records(decoded, f"stdout:{line_number}", preserve_contexts)

    records = []
    known_names = _TEXT_METRICS | {norm_metric_key(key) for key in expected_names}
    for line_number, line in enumerate(lines, 1):
        previous_end = 0
        summaries = list(_SUMMARY_RE.finditer(line))
        matches = [(match, True) for match in summaries]
        matches += [(match, False) for match in _METRIC_RE.finditer(line)
                    if not any(summary.start() <= match.start() < summary.end() for summary in summaries)]
        matches.sort(key=lambda item: item[0].start())
        for index, (match, summary) in enumerate(matches):
            # 脚本自印的论文参考值不能覆盖真实测量。
            prefix = re.split(r"[,;，；]", line[previous_end:match.start()])[-1]
            previous_end = match.end()
            reference_prefix = re.split(r"[,;，；]", line[:match.start()])[-1]
            if _REFERENCE_RE.search(reference_prefix):
                continue
            name = norm_metric_key(match["name"])
            qualified = re.fullmatch(r"(test|eval|evaluation|validation|valid|val|train|training)_(.+)", name)
            if preserve_contexts and qualified:
                prefix += " " + qualified[1]
                name = norm_metric_key(qualified[2])
            if name in {"epoch", "step", "seed", "version"} | _STAT_NAMES:
                continue
            if name not in known_names and (summary or match["sep"] != "="):
                continue
            stage = "final" if re.search(r"\bfinal\b|最终", prefix, re.I) else ""
            split = ""
            if re.search(r"\b(?:test|eval|evaluation)\b|测试|评估", prefix, re.I):
                split, stage = "test", "final"
            elif re.search(r"\b(?:validation|valid|val)\b|验证", prefix, re.I):
                split, stage = "validation", "validation"
            elif re.search(r"\b(?:train|training)\b|训练", prefix, re.I):
                split, stage = "train", "train"
            # Statistics may follow this metric, but never borrow a second
            # metric's uncertainty/count on the same line.
            end = next((other.start() for other, _ in matches[index + 1:]
                        if norm_metric_key(other['name']) not in _STAT_NAMES), len(line))
            suffix = line[match.end():end]
            stats = _statistics(prefix, suffix, match if summary else None)
            records.append({"name": name, "value": metric_number(match["value"]),
                            "unit": "percent" if match["percent"] else "",
                            "split": split, "stage": stage, "source": f"stdout:{line_number}",
                            "source_quote": line.strip(), **stats})
    records = _select_records(records, preserve_contexts)
    errors = [f"{r['name']}: 指标不是有限数值" for r in records if r["value"] is None]
    for record in records:
        errors.extend(_statistic_errors(record))
    return records, errors

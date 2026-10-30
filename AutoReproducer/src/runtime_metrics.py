"""最终运行指标：结构化输出优先，文本回退保留单位和来源。"""
import ast
import json
import math
import re

from src.metric_keys import norm_metric_key
from src.author_comparison_semantics import _exact_repetition_context

METRICS_MARKER = "AUTOREPRO_METRICS:"
NUMBER_PATTERN = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
_METRIC_LABEL = (
    r"(?:macro|micro|weighted)[_\s-]?f1(?:[_\s-]?score)?|"
    r"root[_\s-]?mean[_\s-]?squared[_\s-]?error|"
    r"mean[_\s-]?squared[_\s-]?error|测试集准确率|准确率|精确率|召回率|损失|"
    r"[A-Za-z_][A-Za-z0-9_-]*(?:\s+score)?")
_METRIC_RE = re.compile(
    r"(?<![\w.])(?P<name>" + _METRIC_LABEL + r")"
    r"\.?\s*(?P<sep>[:：=])\s*(?P<value>" + NUMBER_PATTERN + r")(?![\w.])\s*(?P<percent>%)?",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"(?<![\w.])(?P<name>" + _METRIC_LABEL + r")\.?"
    r"(?:\s+(?:of|over|across|from)\s+(?P<runs>\d+)\s+(?:runs|trials|seeds|repeats))?"
    r"\s*[:=]?\s+(?P<aggregation>mean|average|avg|median|best)\s*[:=]?\s*"
    r"(?P<value>" + NUMBER_PATTERN + r")(?![\w.])\s*(?P<percent>%)?", re.I)
_AGGREGATIONS = {'single', 'mean', 'median', 'best', 'unknown'}
_STAT_NAMES = {'mean', 'average', 'avg', 'std', 'stdev', 'stddev', 'median', 'best',
               'runs', 'trials', 'seeds', 'n'}
_RUN_COUNT = r'(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)'
_RUNS_RE = re.compile(r'(?<![\w.,+−–—-])(?:(?P<before>' + _RUN_COUNT + r')\s+'
    r'(?:(?:independent|random|repeated)\s+)?(?:runs|trials|seeds|repeats)|'
    r'(?P<label>runs|trials|seeds|repeats)\s*[:=]\s*(?P<after>' + _RUN_COUNT + r'))\b', re.I)
_RUN_WORDS = re.compile(r'\b(?:runs|trials|seeds|repeats)\b', re.I)
_STD_RE = re.compile(r"(?:\b(?:std(?:dev)?|stdev|standard\s+deviation)\s*[:=]?\s*|"
                     r"(?:±|\+/-)\s*)(?P<std>" + NUMBER_PATTERN + r")(?![\w.])", re.I)
_OTHER_UNCERTAINTY = re.compile(r"\b(?:ci|confidence\s+interval|sem|stderr|standard\s+error)\b", re.I)
_REFERENCE_RE = re.compile(r"论文|声明|参考|目标|\b(?:paper|declared|reference|target)\b", re.I)
_FINAL_STAGES = {"final", "full", "test", "eval", "evaluation"}
_TEXT_METRICS = {"accuracy", "f1_score", "f1_macro", "f1_micro", "f1_weighted",
                 "precision", "recall", "loss", "mse", "rmse", "mae",
                 "r2", "auc", "roc_auc", "mape", "smape", "psnr", "ssim", "bleu", "rouge",
                 "perplexity", "score", "reproduction_score"}
_LOGGER_RUN = re.compile(r'^\s*Run\s+(\d+)\s*:\s*$', re.I)
_LOGGER_ALL = re.compile(r'^\s*All\s+runs\s*:\s*$', re.I)
_LOGGER_FINAL = re.compile(r'^\s*Final\s+(?P<split>train|valid|validation|test)\s*:\s*'
                           r'(?P<value>' + NUMBER_PATTERN + r')(?![\w.])\s*(?P<percent>%)?', re.I)
_SUMMARY_RUN_HEADER = re.compile(
    r'^\s*(?:(?:(?:final|overall)\s+)?(?:results|summary)\s+(?:over|across|from|of)\s+'
    r'(?P<summary_count>\d+)\s+(?:(?:completed|independent|repeated)\s+)?'
    r'(?:runs|trials|seeds|repeats|experiments)|'
    r'[\w.+-]+\s+on\s+dataset\s+[\w.+-]+,\s+in\s+(?P<experiment_count>\d+)'
    r'\s+repeated\s+experiments?)\s*:\s*$', re.I)


def _summary_run_header(line, line_number):
    """A reported summary count, never a configured or independently checked count."""
    match = _SUMMARY_RUN_HEADER.fullmatch(line)
    if not match or _REFERENCE_RE.search(line):
        return None
    count = _run_count(match['summary_count'] or match['experiment_count'])
    if count is None or count < 1:
        return None
    return {'value': count, 'source': f'stdout:{line_number}',
            'source_quote': line.strip(), 'kind': 'reported_summary'}


def _run_count(value):
    try:
        return int(value.replace(',', ''))
    except (TypeError, ValueError, OverflowError):
        return None


def _reported_run_counts(text):
    """Read exact counts without turning bounds, seed IDs or fragments into counts."""
    if _REFERENCE_RE.search(text):
        return set(), bool(_RUN_WORDS.search(text))
    counts, rejected = set(), False
    matches = list(_RUNS_RE.finditer(text))
    for match in matches:
        count = _run_count(match['before'] or match['after'])
        statistics = list(_STD_RE.finditer(text[:match.start()]))
        preceding_statistic = bool(statistics and
            text[statistics[-1].end():match.start()].strip() in {',', ';'})
        if (count is None or not _exact_repetition_context(
                text, match.start(), match.end(), number_follows_label=bool(match['label']),
                preceding_statistic=preceding_statistic)
                or match['label'] and match['label'].lower() == 'seeds'
                and not re.search(r'\b(?:number|count)\s+of\s*$', text[:match.start()], re.I)):
            rejected = True
        else:
            counts.add(count)
    if not matches and _RUN_WORDS.search(text):
        rejected = True
    return counts, rejected


def _bind_summary_runs(records, start, header, line):
    """Only the immediately following, wholly statistical line belongs to a header."""
    rows = records[start:]
    if (not header or header.get('ambiguous') or not rows
            or any(row.get('aggregation') not in {'mean', 'median', 'best'} for row in rows)
            or _REFERENCE_RE.search(line)):
        return []
    counts, rejected_counts = _reported_run_counts(line)
    if rejected_counts:
        return []
    if len(counts) > 1:
        for row in rows:
            row.pop('runs', None)
        return [f"{rows[0]['source']}: Metric line reports conflicting repetition counts"]
    errors = []
    for row in rows:
        if row.get('runs', header['value']) != header['value']:
            row.pop('runs', None)
            errors.append(f"{row['source']}: Summary header and metric report different repetition counts")
        else:
            row['runs'] = header['value']
            row['runs_source'] = dict(header)
    return errors


def _printed_logger_context(line):
    if not line.strip().startswith('Namespace('):
        return None
    fields = {}
    try:
        node = ast.parse(line.strip(), mode='eval').body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace':
            return fields
        for keyword in node.keywords:
            if keyword.arg in ('metric', 'runs'):
                fields[keyword.arg] = ast.literal_eval(keyword.value)
    except (SyntaxError, ValueError, TypeError):
        return {}
    name = {'rocauc': 'roc_auc'}.get(str(fields.get('metric')), norm_metric_key(fields.get('metric')))
    fields['metric'] = name if name in _TEXT_METRICS else None
    if type(fields.get('runs')) is not int or fields['runs'] < 1:
        fields.pop('runs', None)
    return fields


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
    count_context = prefix + ' ' + (summary.group() if summary is not None else '') + ' ' + suffix
    counts, rejected_counts = _reported_run_counts(count_context)
    if len(counts) == 1 and not rejected_counts:
        fields['runs'] = counts.pop()
    std = _STD_RE.search(suffix)
    uncertainty_context = prefix + ' ' + suffix
    if std and _OTHER_UNCERTAINTY.search(uncertainty_context):
        # Authors often label a CI/SEM before the metric rather than after
        # the +/- quantity. Keep that observation without calling it SD.
        fields['reported_uncertainty'] = metric_number(std.group('std'))
        fields['uncertainty_kind'] = ('standard_error' if re.search(
            r'\b(?:sem|stderr|standard\s+error)\b', uncertainty_context, re.I)
            else 'confidence_interval')
        confidence = re.search(r'\b(\d+(?:\.\d+)?)\s*%\s*(?:ci|confidence\s+interval)\b',
                               uncertainty_context, re.I)
        if confidence and 0 < float(confidence[1]) < 100:
            fields['confidence_level'] = float(confidence[1])
    elif std:
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
    if 'reported_uncertainty' in record:
        uncertainty = metric_number(record['reported_uncertainty'])
        if uncertainty is None or uncertainty < 0:
            errors.append('报告误差必须是有限非负数')
            uncertainty = None
        record['reported_uncertainty'] = uncertainty
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
        # Header provenance is derived from this stream's actual text, never
        # accepted as a structured producer's claim about another line.
        record.pop('runs_source', None)
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

    records, errors = [], []
    known_names = _TEXT_METRICS | {norm_metric_key(key) for key in expected_names}
    logger, current_run, all_runs, completed_runs = {}, None, False, {}
    pending_summary = None
    for line_number, line in enumerate(lines, 1):
        summary_context, pending_summary = pending_summary, _summary_run_header(line, line_number)
        if pending_summary is not None:
            if summary_context is not None:
                pending_summary = {'ambiguous': True}
            continue
        line_start = len(records)
        context = _printed_logger_context(line)
        if context is not None:
            logger, current_run, all_runs, completed_runs = context, None, False, {}
            continue
        run_header = _LOGGER_RUN.fullmatch(line)
        if run_header:
            current_run, all_runs = _run_count(run_header[1]), False
        if _LOGGER_ALL.fullmatch(line):
            current_run, all_runs = None, True
        final = _LOGGER_FINAL.match(line) if logger.get('metric') else None
        if final and not _REFERENCE_RE.search(line):
            suffix = line[final.end():]
            # Do not attach a second metric's uncertainty to this bare label.
            others = [match for match in _METRIC_RE.finditer(suffix)
                      if norm_metric_key(match['name']) not in _STAT_NAMES]
            if not others:
                split = {'valid': 'validation'}.get(final['split'].lower(), final['split'].lower())
                stats = _statistics('', suffix)
                if not all_runs and current_run is not None:
                    stats.setdefault('aggregation', 'single')
                    stats.setdefault('runs', 1)
                    if split == 'test' and metric_number(final['value']) is not None:
                        completed_runs[current_run] = completed_runs.get(current_run, 0) + 1
                count = logger.get('runs')
                complete_ids = set(completed_runs)
                if (all_runs and count and len(complete_ids) == count
                        and all(value == 1 for value in completed_runs.values())
                        and complete_ids in (set(range(count)), set(range(1, count + 1)))):
                    stats.setdefault('runs', count)
                records.append({'name': logger['metric'], 'value': metric_number(final['value']),
                    'unit': 'percent' if final['percent'] else '', 'split': split,
                    'stage': 'final' if split == 'test' else split,
                    'source': f'stdout:{line_number}', 'source_quote': line.strip(), **stats})
        previous_end = 0
        summaries = list(_SUMMARY_RE.finditer(line))
        matches = [(match, True) for match in summaries]
        matches += [(match, False) for match in _METRIC_RE.finditer(line)
                    if not any(summary.start() <= match.start() < summary.end() for summary in summaries)]
        matches.sort(key=lambda item: item[0].start())
        for index, (match, summary) in enumerate(matches):
            # 脚本自印的论文参考值不能覆盖真实测量。
            prefix = re.split(r"[,;，；|]", line[previous_end:match.start()])[-1]
            previous_end = match.end()
            reference_prefix = re.split(r"[,;，；|]", line[:match.start()])[-1]
            if _REFERENCE_RE.search(reference_prefix):
                continue
            # Several metrics can share one explicit split label, e.g.
            # "train loss:... acc:... | val loss:... acc:...". Inherit it
            # only within this delimited clause, using the nearest label.
            split_prefix = reference_prefix
            name = norm_metric_key(match["name"])
            qualified = re.fullmatch(r"(test|eval|evaluation|validation|valid|val|train|training)_(.+)", name)
            if preserve_contexts and qualified:
                prefix += " " + qualified[1]
                split_prefix += " " + qualified[1]
                name = norm_metric_key(qualified[2])
            if name in {"epoch", "step", "seed", "version"} | _STAT_NAMES:
                continue
            if name not in known_names:
                continue
            stage = "final" if re.search(r"\bfinal\b|最终", prefix, re.I) else ""
            split = ""
            splits = list(re.finditer(r"\b(?:test|eval|evaluation|validation|valid|val|train|training)\b|测试|评估|验证|训练",
                                      split_prefix, re.I))
            closest_split = splits[-1].group().lower() if splits else ''
            if closest_split in {'test', 'eval', 'evaluation', '测试', '评估'}:
                split, stage = "test", "final"
            elif closest_split in {'validation', 'valid', 'val', '验证'}:
                split, stage = "validation", "validation"
            elif closest_split in {'train', 'training', '训练'}:
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
        errors.extend(_bind_summary_runs(records, line_start, summary_context, line))
    records = _select_records(records, preserve_contexts)
    errors.extend(f"{r['name']}: 指标不是有限数值" for r in records if r["value"] is None)
    for record in records:
        errors.extend(_statistic_errors(record))
    return records, errors

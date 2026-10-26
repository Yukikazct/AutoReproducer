"""Readable report presentation with a hash-bound, unabridged details file."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote

from src.safety.paths import workspace_path


_LINK = re.compile(r'^\[完整依据与执行详情\]\(([^\r\n)]+)\) <!-- sha256:([0-9a-f]{64}) -->$', re.M)


def report_details_bytes(report, report_path):
    """Read only an emitted sibling details file whose bytes match its hash."""
    result = {}
    report_file = Path(report_path)
    if not report_file.is_absolute():
        from src.execution_artifacts import PROJECT_ROOT
        report_file = PROJECT_ROOT / report_file
    for target, digest in _LINK.findall(report):
        name = unquote(target)
        if len(Path(name).parts) != 1 or not name.endswith('_details.md'):
            raise ValueError('报告详情附件路径无效')
        try:
            path = workspace_path(report_file.absolute().parent, name,
                                  'report details attachment', must_exist=True)
        except OSError as exc:
            raise ValueError('报告详情附件缺失或无法读取') from exc
        if not path.is_file():
            raise ValueError('报告详情附件缺失或无法读取')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('报告详情附件哈希不一致')
        result[name] = raw
    return result


def presentation_data(data):
    """Keep interpreted settings and results, moving literal payloads to details."""
    def visit(node):
        if isinstance(node, dict):
            return {key: '' if key in {'quote', 'source_quote', 'value_quote', 'column_quote',
                                      'stdout', 'stderr', 'code'} and isinstance(value, str)
                    else visit(value) for key, value in node.items()}
        if isinstance(node, list):
            return [visit(item) for item in node]
        return deepcopy(node)
    return visit(data)


def readable_text(text, data):
    """Remove empty literal-output scaffolding while retaining explanatory notes."""
    lines, skip_output, skip_collaboration = [], False, False
    for line in text.splitlines():
        if line == '### 执行输出(full)':
            skip_output = True
            final = (data.get('execution') or {}).get('final') or {}
            output_state = '已保存' if final.get('stdout') else '无输出或未记录'
            lines += ['### 执行记录', '', '- **标准输出**: ' + output_state + '。',
                      '完整标准输出、诊断日志和代码保存在详情附件中。', '']
            continue
        if skip_output:
            if line.startswith('## '):
                skip_output = False
            else:
                continue
        if line.startswith('## ') and '协作记录' in line:
            skip_collaboration = True
            messages = (data.get('agent_collaboration') or {}).get('messages') or []
            lines += ['## Agent 协作摘要', '', f'已保存 {len(messages)} 条阶段交接与核查消息，完整记录见详情附件。', '']
            continue
        if skip_collaboration:
            if line.startswith('## '):
                skip_collaboration = False
            else:
                continue
        if (line.strip() in {'>', '**原文依据**'} or line.startswith('- **原文上下文**:')
                or line.strip().startswith('- **原文**:') or line.startswith('- **发现来源**:')
                or line.startswith('- **仓库关联证据**:')):
            continue
        # Source identifiers remain visible when the corresponding quotation is
        # moved, including references emitted as "> source_id: quote".
        if re.fullmatch(r'>\s*[^:：]+[:：]\s*', line):
            continue
        line = line.replace('**计划引用原文**', '**计划依据位置**')
        lines.append(line)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).replace(
        '# 论文复现与优化报告', '# 论文复现实验报告', 1)


def save_report_details(full_report, data, report_path):
    """Retain raw quotation values even where the legacy audit used excerpts."""
    values = {}
    def collect(node, path='data'):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {'quote', 'source_quote', 'value_quote', 'column_quote'} and isinstance(value, str) and value:
                    values.setdefault(value, []).append(path + '.' + key)
                else:
                    collect(value, path + '.' + str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                collect(value, f'{path}[{index}]')
    collect(data)
    if values:
        full_report += '\n\n## 完整引用字段\n\n原始引用字段逐项保留，定位键对应保存的结构化结果。\n'
        for value, locations in values.items():
            fence = '`' * max(3, 1 + max((len(run) for run in re.findall(r'`+', value)), default=0))
            full_report += '\n### ' + ', '.join(locations) + '\n\n' + fence + 'text\n' + value + '\n' + fence + '\n'
    collaboration = data.get('agent_collaboration') or {}
    if collaboration.get('messages'):
        raw_collaboration = json.dumps(collaboration, ensure_ascii=False, indent=2)
        fence = '`' * max(3, 1 + max((len(run) for run in re.findall(r'`+', raw_collaboration)), default=0))
        full_report += ('\n\n## 完整协作记录\n\n保存本次报告生成时的完整协作快照；消息本身不构成核验结论。\n\n'
                        + fence + 'json\n' + raw_collaboration + '\n' + fence + '\n')
    path = Path(report_path).absolute()
    if not Path(report_path).is_absolute():
        from src.execution_artifacts import PROJECT_ROOT
        path = PROJECT_ROOT / report_path
    name = path.stem + '_details.md'
    target = workspace_path(path.parent, name, 'report details attachment')
    target.parent.mkdir(parents=True, exist_ok=True)
    from src.source_repair_evidence import _replace_bytes
    raw = full_report.encode('utf-8')
    _replace_bytes(target, raw)
    return '[完整依据与执行详情](' + quote(name, safe='') + ') <!-- sha256:' + hashlib.sha256(raw).hexdigest() + ' -->'

"""指标键归一：判定层与展示层共用的唯一规则。

为什么需要单独一个模块：论文声明的指标名由 LLM 从论文里抽取（`MSE`、
`F1 Score`、`Test-Accuracy` 各种写法都有），而运行输出里的指标名由
`ResultValidator._extract_metrics` 按正则归一成小写（`mse`、`f1_score`）。
两边若各自实现比对规则，就会出现"判定说匹配、报告表格里却排成两行"的
自相矛盾——实测已发生过一次（判定层 2.6% 的差异被误报成未复现）。

所以归一规则只写在这里，判定层（`result_validator`）与展示层
（`report_generator` 的指标对比表）都从这里取。
"""
import re

_SEP_RE = re.compile(r"[\s\-]+")


def norm_metric_key(key) -> str:
    """指标键归一：大小写、空白、连字符差异不构成"不同指标"。

    只折叠写法差异，不合并真正不同的指标——`rmse` 与 `mse` 归一后仍不同。
    """
    return _SEP_RE.sub("_", str(key).strip().lower())

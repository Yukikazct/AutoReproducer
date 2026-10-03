# 运行结果与论文数值核验（Issue #18）

本轮修复对应 [Issue #18](https://github.com/Yukikazct/AutoReproducer/issues/18) 和
[实施方案](repository_reproduction_plan.md) 的验证 P0。修改只涉及结果判定与指标证据，仓库执行链路另由 Issue #16/#17 实施。

## 判定规则

- 任一当前必需步骤非零退出、超时、取消或明确失败，不能通过数值核验。自愈后成功重跑的旧失败仅作为历史保留。
- 未提供最终执行成功证据，结果为无法核验；只有 smoke/check 完成时标记“仅冒烟通过”。
- 完整执行成功后，所有声明/必需指标必须存在且为有限数值。参考值无效、缺失指标或单位不一致均不能通过。
- 每项指标的相对差异不超过 5% 才通过；参考值为零时实测值也须为零。
- 只按显式 `percent` / `fraction` 单位换算，不按数值大小推断百分比。MSE=50 与 MSE=0.5 相差 100 倍。
- 没有论文参考数值时保留运行指标，显示“无法核验”。语料库的 `reproduction_score` 不自动充当论文性能指标。
- 模型仅解释差异；模型结论和 API 故障不能改变确定性判定。数值核验通过不代表已经核实论文全部方法、数据和实验配置。

报告同时显示执行级别、逐项差异及指标单位、数据划分、阶段和来源。旧报告保持原有内容；新运行生成新判定。

## 最终指标协议

生成脚本在评估完成后打印 `AUTOREPRO_METRICS: ` 加单行 JSON：

```python
import json

# measured_mse 必须来自本次评估。
payload = {
    "version": 1,
    "metrics": [
        {"name": "mse", "value": measured_mse, "unit": "",
         "split": "test", "stage": "final"}
    ],
}
print("AUTOREPRO_METRICS: " + json.dumps(payload, allow_nan=False))
```

只使用最后一个结构化消息；结构化消息损坏时拒绝核验，不回退到早先训练日志。
执行器也可通过最终阶段的 `metrics` 映射或 `metric_records` 列表提供同样的证据。
暂未输出结构化消息的旧脚本使用文本回退，支持符号和科学计数法、同阶段最后一次测量；明确的最终/测试输出优先，论文声明行排除。

论文侧可保留现有数值字典，并另外明确单位：

```json
{
  "metrics": {"accuracy": 0.85},
  "metric_units": {"accuracy": "fraction"}
}
```

需要固定数据划分或阶段时，也可将单个参考指标写为
`{"value": 0.5, "unit": "", "split": "test", "stage": "final"}`。
单位未知时留空，双方都未标注时只比较原始数值；任何跨单位换算都须先定义单位。

## 验证与页面测试

`tests/test_result_validation_gate.py` 覆盖 Astra 记录的五个反例、非有限值、显式单位、执行层级、结构化消息、修复历史、模型不能改判、网页失败报告和实际子进程结果。

```bash
.venv/bin/python -m pytest tests/test_result_validation_gate.py -q
```

页面可重新运行模拟示例，检查报告中的“论文数值核验通过”“完整执行已完成”和“实际指标证据”；模拟结果仍为模拟数据。
上传未给出参考指标数值的 PDF 时，实际指标和图片应继续显示，数值结论应为“无法核验”。
失败运行即使已经打印数值，也应显示执行失败，并跳过优化。历史报告不会自动重算，须新建一次运行测试。

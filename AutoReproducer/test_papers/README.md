# 独立测试 PDF

`dgc_2102.10739_holdout_20261013.pdf` 是论文 *Dissecting the Diffusion Process in Linear Graph Convolutional Networks* 的 arXiv PDF，来源 https://arxiv.org/pdf/2102.10739。可在网页选择“上传 PDF”进行测试；使用真实 API 会产生模型服务费用，并安装作者代码的依赖。

2026-10-13 的冻结版本已通过真实 DeepSeek API 和网页后端 PDF 入口完成作者 Cora 代码运行。只提供 PDF，系统自行提取作者仓库与运行命令。作者程序完成 10 次内置运行、1000 次优化器更新、退出码 0。

这份 PDF 对应此前的冻结版本。后续浏览器测试又触发产品修复，DGC 已退役为回归样本。

`digcn_holdout_20261013_05.pdf` 是最新独立验收论文 *Digraph Inception Convolutional Networks* 的 NeurIPS 官方 PDF。修复后只通过真实浏览器上传该 PDF，系统自行提取仓库和命令，完成作者 Cora-ML 实验：20 次原生运行、10,049 次优化器更新、退出码 0。原始输出测试准确率为 `0.7719 ± 0.0044`，网页报告显示实际数值。

来源与摘要见同目录 `.provenance.json`；详细范围、限制和证据见 [测试跟进](../docs/test_followup_20261013.md)。运行后论文数值比较尚未接入。

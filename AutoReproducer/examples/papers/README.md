# 已下载论文 PDF 样本

收录本轮排查、候选选择和真实网页测试中已下载的论文原始 PDF。可直接从 `pdf/` 选择文件上传到网页；每份文件的来源、页数、字节数和 SHA-256 见 [manifest.json](manifest.json)。

共 21 个本地输入文件，按 SHA-256 去重后保存 **20 份 PDF**，合计 **35.93 MiB**。ReZero 的两份相同输入仅保留一份，原文件名映射保存在清单中；JacobiConv 的 v1/v2 和 ReZero 的 arXiv/出版版本内容不同，分别保留。

2026-10-12 又归档了 [23 份候选 PDF](candidates_20261012/README.md)，合计 41.55 MiB；来源、原文仓库证据与当前试验状态见该目录的 [独立清单](candidates_20261012/manifest.json)。两个目录共保存 43 份 PDF。CCA-SSG 已完成原始作者命令，见[执行验收](../../docs/ccassg_author_execution_20261012.md)；论文数值尚未独立核验。资料收录不代表复现通过。

这些文件是候选或回归资料；文件名中的 `holdout`、`independent` 或 `finaltest` 记录历史用途，不代表当前未见样本或复现通过。完整验收状态见 [独立新 PDF 验收记录](../../docs/independent_pdf_acceptance.md)，最新阻塞见 [Issue #30](https://github.com/Yukikazct/AutoReproducer/issues/30)。

清单中的 `download_record` 来源由原下载记录及文件哈希核对；`existing_project_reference_no_download_receipt` 表示来源链接来自已有项目记录，但原始下载回执缺失，最终跳转地址和下载时间不作推断。

## 论文样本

| PDF | 论文 | 页数 | 来源 |
|---|---|---:|---|
| [acm_gnn_holdout_08_20261011.pdf](pdf/acm_gnn_holdout_08_20261011.pdf) | Revisiting Heterophily For Graph Neural Networks | 41 | [原来源](https://raw.githubusercontent.com/SitaoLuan/ACM-GNN/main/Revisiting_Heterophily_For_Graph_Neural_Networks_updated.pdf) |
| [chebnetii_independent_20261011_function.pdf](pdf/chebnetii_independent_20261011_function.pdf) | Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited | 20 | [原来源](https://arxiv.org/pdf/2202.03580) |
| [dagnn_independent_20261011_b9d5219.pdf](pdf/dagnn_independent_20261011_b9d5219.pdf) | Towards Deeper Graph Neural Networks | 11 | [原来源](https://arxiv.org/pdf/2007.09296) |
| [deepxde_1907.04502.pdf](pdf/deepxde_1907.04502.pdf) | DeepXDE: A Deep Learning Library for Solving Differential Equations | 21 | [原来源](https://arxiv.org/pdf/1907.04502) |
| [dlinear_2205.13504v3_finaltest_20261011.pdf](pdf/dlinear_2205.13504v3_finaltest_20261011.pdf) | Are Transformers Effective for Time Series Forecasting? | 15 | [原来源](https://arxiv.org/pdf/2205.13504v3) |
| [dropedge_1907.10903_holdout_07_20261011.pdf](pdf/dropedge_1907.10903_holdout_07_20261011.pdf) | DropEdge: Towards Deep Graph Convolutional Networks on Node Classification | 18 | [原来源](https://arxiv.org/pdf/1907.10903) |
| [fsgnn_2105.07634_holdout_04.pdf](pdf/fsgnn_2105.07634_holdout_04.pdf) | Improving Graph Neural Networks with Simple Architecture Design | 10 | [原来源](https://arxiv.org/pdf/2105.07634) |
| [gcnii_2007.02133_holdout_05.pdf](pdf/gcnii_2007.02133_holdout_05.pdf) | Simple and Deep Graph Convolutional Networks | 13 | [原来源](https://arxiv.org/pdf/2007.02133) |
| [gdc_independent_20261011_ec4b097.pdf](pdf/gdc_independent_20261011_ec4b097.pdf) | Diffusion Improves Graph Learning | 22 | [原来源](https://arxiv.org/pdf/1911.05485) |
| [ggcn_independent_20261011_bc20605.pdf](pdf/ggcn_independent_20261011_bc20605.pdf) | Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks | 25 | [原来源](https://arxiv.org/pdf/2102.06462) |
| [gnn_lfhf_holdout_06_20261011.pdf](pdf/gnn_lfhf_holdout_06_20261011.pdf) | Interpreting and Unifying Graph Neural Networks with An Optimization Framework | 12 | [原来源](https://raw.githubusercontent.com/zhumeiqiBUPT/GNN-LF-HF/master/resources/Paper.pdf) |
| [gprgnn_2006.07988_holdout_03.pdf](pdf/gprgnn_2006.07988_holdout_03.pdf) | Adaptive Universal Generalized PageRank Graph Neural Network | 24 | [原来源](https://arxiv.org/pdf/2006.07988) |
| [jacobi_independent_20261011_0b2a771.pdf](pdf/jacobi_independent_20261011_0b2a771.pdf) | How Powerful are Spectral Graph Neural Networks | 19 | [原来源](https://arxiv.org/pdf/2205.11172v1) |
| [jacobi_v2_independent_20261011_0b2a771.pdf](pdf/jacobi_v2_independent_20261011_0b2a771.pdf) | How Powerful are Spectral Graph Neural Networks | 22 | [原来源](https://arxiv.org/pdf/2205.11172v2) |
| [neural_ode_input_1806.07366.pdf](pdf/neural_ode_input_1806.07366.pdf) | Neural Ordinary Differential Equations | 18 | [原来源](https://arxiv.org/pdf/1806.07366) |
| [rezero_2003.04887.pdf](pdf/rezero_2003.04887.pdf) | ReZero is All You Need: Fast Convergence at Large Depth | 14 | [原来源](https://arxiv.org/pdf/2003.04887v2) |
| [rezero_published_2021_protocol.pdf](pdf/rezero_published_2021_protocol.pdf) | ReZero is All You Need: Fast Convergence at Large Depth | 10 | [原来源](https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf) |
| [sgc_1902.07153_holdout_01.pdf](pdf/sgc_1902.07153_holdout_01.pdf) | Simplifying Graph Convolutional Networks | 14 | [原来源](https://arxiv.org/pdf/1902.07153) |
| [ssgc_holdout_02.pdf](pdf/ssgc_holdout_02.pdf) | Simple Spectral Graph Convolution | 16 | [原来源](https://unsworks.unsw.edu.au/bitstreams/98ccc499-d878-4cd0-b6e9-b626d133a788/download) |

## 错误候选留档

一次 AirGNN 候选下载使用了错误的 arXiv 编号，实际得到 medspaCy 论文。该原文件保存为 [medspaCy 错误候选](rejected/medspacy_2106.07799_wrong_airgnn_candidate.pdf)，清单角色为 `rejected_wrong_paper`，不用于 AirGNN 复现验收。

## 文件校验

在本目录运行以下 PowerShell 命令，逐个检查归档字节数与 SHA-256：

```powershell
$manifest = Get-Content -LiteralPath ./manifest.json -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($paper in $manifest.papers) {
    $file = Get-Item -LiteralPath $paper.file
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($file.Length -ne $paper.bytes -or $hash -ne $paper.sha256) {
        throw "PDF 校验失败：$($paper.file)"
    }
}
Write-Output "全部 PDF 校验通过：$($manifest.papers.Count) 份"
```

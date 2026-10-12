# GNN-LF/HF 仓库定位与真实执行回归

用户上传的论文是 *Interpreting and Unifying Graph Neural Networks with An Optimization Framework*。PDF 第 9 页的 DGL、DropEdge 链接用于基线实现，发现流程却优先选中 DGL 并停止标题搜索。导出阶段又因 `.gitmodules` 直接拒绝，备用模型候选没有进入执行流程。

修复后，只有明确的作者代码声明和用户选择优先于搜索。普通链接保留为线索，模型可选择搜索返回的实现；执行流程最多依次核验六个候选。在默认代码执行目标中，论文与方法匹配即可运行，不要求先证明作者归属。子模块在匹配核验后按固定 gitlink commit 导出，记录嵌套来源与文件哈希。

## 2026-10-12 的真实回归结果

这是针对已经暴露问题的 PDF 的回归，不能作为新的独立论文验收。调用生产资源发现、仓库核验、规划、依赖准备、执行及报告流程；使用真实模型、网络、PDF 和作者仓库，没有指定仓库或手动修改运行命令。此回归直接调用后端，未重复浏览器上传测试。

| 项目 | 结果 |
| --- | --- |
| PDF | `gnn_lfhf_holdout_06_20261011.pdf` |
| PDF SHA-256 | `862f95d01b672cbbe41f0455f70196438c18b1dbfb29206016351b35c49fda05` |
| 自动选择仓库 | https://github.com/zhumeiqiBUPT/GNN-LF-HF |
| 固定 commit | `cbc723a264e6334e81344a68720e72dd181d3c12` |
| 发现链 | Papers with Code → GitHub 搜索 → 模型选择 |
| README 命令 | `python main.py --dataset=cora -t=1 -f=0 --device=0` |
| 实际训练 | 作者脚本的 10 次 Cora 训练，累计 14,068 次优化器更新 |
| 运行环境 | Windows、Python 3.11、PyTorch 2.5.1+cu121 |
| 自然退出 | 退出码 0，无超时，所属进程已结束 |
| 训练命令耗时 | 173.031 秒 |
| 总流程耗时 | 513.75 秒，包含首次依赖准备 |
| 数据 | 使用固定作者仓库内的 Cora 数据，逐文件哈希核验 |
| 兼容修复 / 缺包重试 | 本次均为 0 |
| 产品结论 | `author_code_completed`，`is_reproduced=null` |

作者输出 `Test: ACC: 83.25 ± 0.34%`。该数值是作者程序输出，未进行独立指标计算或论文实验协议验收，不能据此宣布整篇论文复现成功。运行目录为 `data/runs/repository_46d09b286e8046efb7f8cdb9c9734182`，本地结果为 `data/runtime/gnn_lfhf_service_regression_20261012.json`。

新增测试覆盖普通框架链接不阻止搜索、发现备用仓库进入实际服务核验、PDF 链接作者归属未知时允许匹配实现执行，以及子模块固定版本导出和离线重用。明确不同的方法、框架依赖、虚构原文引用和未自然结束的进程仍不能获得匹配实现或运行完成结论。

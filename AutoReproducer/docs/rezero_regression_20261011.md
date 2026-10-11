# ReZero 开发回归记录（2026-10-11）

本样本已参与实现与修复，只能用于回归，不能作为未见 PDF 的最终验收。

原始 PDF：<https://arxiv.org/pdf/2003.04887v2>，14 页，SHA-256 `54eea1741fad2e7a20e52f55c64b34977f1206c5e3c8e14804ea5385c299c070`。真实网页上传后，原文作者链接定位 `majumderb/rezero`，两跳作者 README 关系定位 `tbachlechner/ReZero-Superconvergence`，固定训练版本 `6c0212669ac8c23d3db6f2b99255bcf6e3c5e6e6`。

选定范围为 CIFAR-10 完整实验：50,000 训练样本、10,000 测试样本，45 轮、batch 512、4,410 次更新，固定 seed 6892。原始作者模型和调度器保持不变，执行前冻结参考值 94.00%。

实际运行目录 `data/runs/repository_9a20c2821541421abd0c72b5797cb2d8`。训练用时 1,961.438 秒；最佳 checkpoint 在第 44 轮，独立重载后完整测试准确率 **93.46%**，交叉熵 **0.2580239795687168**。训练协议与独立指标核验通过，数值低于固定参考，结果为 `reference_not_met`、`is_reproduced=false`。不修改门槛，不换种子重跑来覆盖本次结果。

实时记录 `data/runtime/progress_1791687569924.jsonl` 显示训练和独立评估返回、进程清理完成后才确认结束。训练中与结束后的页面检查分别保存在 `rezero_running_page_acceptance_20261011.json` 和 `rezero_completed_page_acceptance_20261011.json`；最终无活跃运行或后台 worker。执行完成与数值通过是两个独立结论。

数据来源备注：实际 CIFAR-10 压缩包从 `https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz` 获取，大小 170,498,071 字节，MD5 `c58f30108f718f92721af3b95e74349a`、SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`。下载凭据保存在 `data/runtime/rezero_cifar10_download_20261011.json`。本次冻结清单使用修复前的缓存来源字段，误把规范源 URL 当作实际下载源；保留原清单并在此披露。后续代码已把规范源、真实下载源和缓存命中分开记录。

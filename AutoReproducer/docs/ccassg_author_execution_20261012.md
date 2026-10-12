# CCA-SSG 作者仓库代码执行验收

2026-10-12，修复后更换未参与修复的论文，通过真实网页上传 PDF，完成原作者 README 的 Cora 命令。测试期间没有修改产品代码；没有手工覆盖模型选择的仓库或实验命令。

| 项目 | 证据 |
|---|---|
| 论文 | From Canonical Correlation Analysis to Self-supervised Graph Neural Networks |
| 原 PDF | <https://arxiv.org/pdf/2106.12484>，24 页，6,008,518 字节 |
| PDF SHA-256 | `be3dfcecb83ac87ca01131895b08385383a9f428d6b5dafb318788d864e99855` |
| 固定作者仓库 | <https://github.com/hengruizhang98/CCA-SSG/tree/ce5b2c0ad4f05811008559f66e5e114dbd71a82b> |
| 冻结产品指纹 | `1b6f1a3af95f8136b6f0aa382b3feb4c73bee239ce0d4c70a064a32456f29879` |
| 网页任务 | `progress_1791788848616.jsonl` |
| 运行标识 | `repository_3e3ee28e63ab44d882bd61e31dac039f` |
| 实际训练 | 50 轮 CCA-SSG 训练及 2000 轮线性评估，2050 次优化更新 |
| 完成判定 | 原命令自然返回，退出码 0；全部所属进程结束 |
| 修复与重试 | 本次没有运行中补包或类型修复，原仓库文件保持不变 |
| 网页结论 | `COMPLETED / author_code_completed`，论文数值尚未核验 |

系统根据 README 选择原命令：

```text
python main.py --dataname cora --epochs 50 --lambd 1e-3 --dfr 0.1 --der 0.4 --lr2 1e-2 --wd2 1e-4
```

实际 Windows CPU 环境使用 Python 3.11.9、torch 2.5.1+cpu、DGL 1.1.2、NumPy 1.26.4、SciPy 1.14.1。原生运算检查通过，原作者代码自行下载并加载 DGL Cora 数据。日志报告 2708 节点、10556 边、1433 特征、7 类，训练/验证/测试样本为 140/500/1000。

程序报告线性评估准确率 `0.8170`。该数值及数据统计是原程序输出；完整数据来源、文件哈希和论文表格数值没有独立核验，本次只证明所选原始命令可运行完成。

真实网页在完成时显示“所选仓库代码已运行完成”和“论文数值尚未核验”。全部执行 invocation 的开始与结束成对，终态活动 worker、实验与执行进程均为零；产品文件及固定仓库源码 SHA-256 再次核对通过。精选证据见 [JSON 记录](ccassg_author_execution_20261012.json)，原 PDF 见[候选目录](../examples/papers/candidates_20261012/ccassg.pdf)。

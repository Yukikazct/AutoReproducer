# DLinear：完整实验重跑步骤

2026-10-07 新运行 `repository_e9408141e3dd44319e854bbdf4a70c0c` 已通过生产网页后台入口完成真实多 Agent 分析、作者完整训练与结果摘要解释。四阶段各调用一次，训练后的解释调用一次，deepseek-chat 实际共 **5 次、零修正重试**；`analysis_status=completed`。16 个真实公开来源、113 个固定源码文件、真实数据、完整运行协议及独立复算均核验通过，`validation.status=reproduced`。详见[新完整报告](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/report.md)、[实际分析记录](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/repository_analysis.json)和[结果解释](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/result_analysis.json)。

本次重新训练第 7 轮按官方 patience 3 早停，MSE 0.3841444、MAE 0.4047131，与[旧版单次 API 基线报告](../data/runs/repository_66efc14890e34cb196a07a7d2bcb78fe/report.md)一致；旧记录保留。完整结果见[复现结果](dlinear_reproduction_result.md)。新流程先分析公开论文、固定作者源码和依赖；此前预分析拒绝的诊断记录也保留，失败不会被跳过以启动训练。

范围固定为论文 Table 2 的 DLinear / ETTh1 多变量、输入 336 步、预测 96 步。[作者确认只使用一个 seed](https://github.com/cure-lab/LTSF-Linear/issues/33#issuecomment-1331937601)，原入口固定 seed 2021。训练保留最多 10 轮、batch 32、学习率 0.005，以及验证集早停和最优 checkpoint；官方早停结束即完成协议，不要求人为跑满 10 轮。

## 命令行完整训练

进入 `AutoReproducer/`，使用 Python 3.11 或 3.12。下面先给出无需 API 的作者完整训练命令：

```bash
python scripts/reproduce_repository.py --profile dlinear_etth1_reference
```

不指定 `--profile` 时也默认执行完整实验。当前预设使用本地 CPU，不依赖 Docker；运行时自动准备固定源码、真实数据和隔离兼容依赖，通过健康检查后执行官方 `run_longExp.py` 完成训练与最终 test。Store 启动环境或应用依赖不兼容时自动切换到标准 CPython 3.11/3.12 环境。源码坏对象和数据坏缓存在线按原 SHA 重新获取；默认公共依赖镜像不可用时回退官方源，冻结版本不变。

源码、数据、公开论文/作者评论和兼容依赖缓存均已准备且希望禁止在线获取时，加 `--offline`。运行环境恢复也不会下载新解释器，应用依赖仅使用本地 wheel；冻结训练依赖必须已有健康缓存，缺失或损坏时停止。明确启用在线分析仍会访问 LLM API。离线缓存无法核验时停止，不使用合成数据。

测试当前多 Agent 分析、完整训练和经用户允许的结果摘要分析时运行：

```bash
python scripts/reproduce_repository.py --profile dlinear_etth1_reference --llm-review --analysis-mode multi_agent --result-review
```

该命令复用 `LLM_BASE_URL`、`LLM_MODEL` 和已有 API 配置；未设置 `LLM_API_KEY` 时通过隐藏输入读取，不要将密钥写进命令或报告。`--llm-review` 启用默认 `multi_agent` 模式：PaperReader、ResourceFinder、EnvBuilder、Verifier 依次分析真实公开论文及固定作者源码，全部通过后才开始训练。

`--result-review` 明确允许训练后把本次 MSE、MAE、已完成轮数、协议核验与独立复算是否通过的摘要发给 API；不发送工作区路径、原始日志、数据文件或 checkpoint。不允许发送结果摘要时去掉此参数，仍可执行四阶段公开分析与训练；单独使用 `--result-review` 而未加 `--llm-review` 会被 CLI 拒绝。

四阶段均一次通过且结果摘要分析通过时是 **4 + 1 = 5 次调用**。每个预分析阶段最多允许一次有原文依据的修正重试，调用数可能增加；失败也可能提前停止。以报告和 `total_llm_calls` 的实际调用计数为准，不固定显示 5 次。API 解释不覆盖本地产物和确定性验收结论；结果摘要解释失败时，已完成训练的验收结果保留，另记 `analysis_status=result_analysis_failed`。

GitHub REST 限额触发时，两个固定作者评论会自动从各自官方 issue 网页恢复；原始正文和作者成员身份必须一致，原 HTML 与评论 JSON 的哈希保留在公开来源 manifest。REST 诊断和网页恢复信息只留在本地证据记录，四阶段分析继续读取完整真实来源。

本次真实成功记录恰好使用 5 次：`repository_analysis.json` 为4次且各阶段 `attempt=1`，`result_analysis.json` 为1次，最终结果与审计也均记录5次。报告中的参数来源区分论文原文、作者实验脚本和代码设置/默认值；作者依赖清单与单seed说明不被当作论文原环境或多seed统计已验证的证据。

CLI 退出码或 `state=COMPLETED` 只说明流程已结束。检查本项数值复现时还应确认 `validation.status=reproduced`、协议核验和独立指标复算均通过；检查完整分析流程时另确认四阶段均接受、结果分析接受且 `analysis_status=completed`。

可选 `--prepare-only` 只导出源码、校验数据和保存计划，不安装训练依赖或训练模型；结果是 `prepared`，不是完整训练结果。`dlinear_etth1_smoke` 保留为开发诊断预设，不是当前默认交付目标。

## 网页重跑

```bash
python -m streamlit run app.py
```

1. 当前网页默认真实模式；保持“🧪 Mock模式（无需API）”关闭。保持“🐳 Docker 沙箱执行（真实模式）”关闭，仓库预设使用本地 CPU。
2. “输入方式”选“官方仓库预设”，选择默认“DLinear · ETTh1 · 完整实验（作者训练协议）”，无需上传 PDF。
3. 保持“仅准备代码、数据和命令（不训练）”未勾选。当前“使用真实多 Agent 分析论文、仓库和环境”默认勾选，填写已有 API 配置和密钥；只运行作者实验时可取消此选项。
4. 需要结果解释时，自行勾选“允许 API 分析本次指标、轮数和核验状态摘要”，该选项默认不勾选，发送范围与 CLI `--result-review` 一致。
5. 点击运行。先查看四阶段的原文引用和接受状态；若预分析拒绝，查看失败原因，不应出现训练。分析通过后检查真实训练日志、官方早停、唯一最终 test 的 `mse:..., mae:...`，以及实际参数、协议核验和独立指标复算。

网页与CLI使用相同 `Orchestrator.run` 路径。成功判定同时要求源码/数据校验通过、所有必需步骤成功、完整作者训练完成、产物齐全、独立指标复算通过，以及两个指标均在既有 **5% 相对容差** 内。该容差是项目验收约定，不是论文公布的阈值。失败、指标缺失或NaN/Inf不会显示数值复现通过。

流水线按实际阶段顺序显示，训练前证据预审与训练后本地核验各有独立状态；重试不会覆盖其他阶段。环境计划完成仅表示计划准备好，依赖实际安装发生在执行作者实验阶段。未启用阶段标记跳过，失败后未执行阶段标记未执行。进度达到 100% 表示所有启用阶段已完成，数值复现结论仍以验收结果为准。

实时输出窗口固定为 240 像素高，可滚动查看；完整日志和报告产物继续保存。智能优化选项当前禁用，仅保留 `enable_optimization=False` 接口；请求 `True` 也不会执行优化，返回尚未开放。

## 查看本次证据

每次运行保存至新的 `data/runs/repository_<id>/`，具体目录由CLI输出或报告给出：

- `repo/`、`repository.json`：固定作者树、commit和源码文件SHA，不带`.git`或历史训练产物。
- `dataset.json`：真实ETTh1来源、17420行、大小与SHA-256。
- `experiment_spec.json`、`execution_plan.json`、`environment.json`：冻结目标、真实命令和兼容环境。
- `protocol_verification.json`：实际Namespace、seed、8209/2785/2785窗口、连续epoch和官方早停、checkpoint/预测数组核验。
- `metrics.json`、`independent_metrics.json`：最终test指标与从真实标签/预测数组独立复算的结果。
- `public_sources.json`、`public_source_manifest` 指向的缓存 manifest：多 Agent 分析所用公开原文、URL、固定源码版本、内容哈希与定位。
- `repository_analysis.json`：四阶段公开分析、原文引用、逐阶段实际调用计数和接受状态；失败时仍保存已完成部分与 `gate.reason`。
- `result_analysis.json`：允许结果摘要分析后保存解释或失败诊断；结果摘要解释状态与本地复现判定分别记录。
- `llm_analysis.json`：旧版 `public_protocol` 单次公开协议解析记录，历史基线使用这一方式。
- `result.json`、`report.md`：最终状态、指标差异与真实实验图。

源码固定为 `cure-lab/LTSF-Linear@0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`。数据SHA-256为 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`。作者产物位于 `repo/checkpoints/<setting>/checkpoint.pth` 和 `repo/results/<setting>/pred.npy`，本项预测shape为 `(2785,96,7)`。原作者的`metrics.npy`和`true.npy`保存被注释，本项目从原始真实数据重建测试标签独立复算。

报告里的图片引用同目录下的 `<报告文件名去掉.md>_assets/` 图片资产目录。网页分享或离线查看时使用“⬇️ 下载报告和图片（ZIP）”，解压后保留 Markdown 与该目录的相对位置。手动复制报告时也要一起复制图片资产目录；“⬇️ 下载本次复现报告”仅下载 Markdown，单独移动它无法显示这些图片。

本次运行可直接下载[报告与两张真实实验图 ZIP](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/report_with_figures.zip)。其中预测对照图和训练/验证损失图均来自本次真实运行产物。

## Windows与GPU

10 月 7 日历史验收环境为 macOS ARM64、Python 3.12.2、PyTorch 2.5.1、NumPy 1.26.4、Pandas 2.2.3、scikit-learn 1.5.2、Matplotlib 3.9.2。作者原依赖声明 PyTorch 1.9.0，环境差异随报告记录，算法源码保持固定版本。

2026-10-10 已在 Windows Store 启动环境完成自动恢复和完整 CPU 重跑：运行 `repository_b707311a3bb84f9db886af456a3780b5` 自动切换 `.venv-py312`、恢复限额作者评论、从官方源完成冻结依赖安装，最终 MSE **0.3841444**、MAE **0.4047131**，验收为 `reproduced`。四阶段公开分析实际调用 **4 次**，未启用结果摘要解释；见[报告](../data/runs/repository_b707311a3bb84f9db886af456a3780b5/report.md)与[结果](../data/runs/repository_b707311a3bb84f9db886af456a3780b5/result.json)。

Windows 可直接运行现有 CPU 预设，所需独立依赖会按 Python ABI、OS 和架构自动准备与检查。当前预设通过 `CUDA_VISIBLE_DEVICES` 固定 CPU，仓库 Docker 路径尚不支持。DLinear CUDA 未在该验收中测试。

作者说明硬件与PyTorch版本可能影响数值。重跑应比较固定协议、真实指标和既有容差，不承诺各平台得到逐位相同的MSE/MAE。

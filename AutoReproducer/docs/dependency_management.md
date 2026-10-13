# 依赖环境管理

## 应用与测试依赖

`requirements.txt` 是应用依赖的唯一清单，包含网页、PDF 解析、文件锁，以及主进程直接使用的 packaging、NumPy 和 Pillow。`requirements-test.txt` 引用它，再添加 pytest 和测试绘图所需的 matplotlib。PyTorch、SciPy 等论文运行包仍按实验安装到托管环境。

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt  # 开发与测试环境
python -m pip check
```

自动准备 CPython 3.11/3.12 时从源码位置读取同一份生产清单，安装与健康检查同步使用其版本要求，包含上限约束。恢复入口导入阶段只依赖标准库；缺少 packaging 等应用依赖时可以继续进入独立环境修复。测试依赖不会被自动安装到应用恢复环境。

## 论文运行环境

当前作者仓库与固定预设入口均设置 `dependency_health_check=False`。默认仍按 Python ABI、操作系统、架构和归一化依赖清单创建隔离缓存；相同清单的 `.ready` 环境可以复用，执行期间持有缓存锁。但这些入口不执行下文的缓存原生健康检查、跨清单增量共享或安装前自动回收。需要回收空间时，请使用 `deps-prune` 预览并执行。此开关不影响上文的应用依赖检查与兼容 Python 环境准备。

按完整 requirements 清单创建环境时，即使只是多一个小包，也可能重新安装一整套 PyTorch；不同 Python 版本和 CPU／CUDA 环境的缓存还会继续累积。2026-10-12 的 SGC 回归曾在环境准备阶段遇到 `No space left on device`，训练尚未启动。

### macOS 与 Windows/Linux 的安装包版本

作者仓库的依赖规划、安装校验和运行时缺包修复共用 `src/dependency_versions.py` 的版本选择。macOS 使用 `torch==2.5.1`、`torchvision==0.20.1`、`torch-scatter==2.1.2`、`torch-sparse==0.6.18`；Windows/Linux CPU 构建保留对应的 `+cpu` / `+pt25cpu` 后缀，CUDA 构建保留 `+cu121` / `+pt25cu121`。Neural ODE 的固定 CPU 预设采用相同的 PyTorch 选择规则。

macOS 图计算包仍从 PyG 的 `torch-2.5.0+cpu.html` 索引获取；索引名中的 `+cpu` 不代表 macOS wheel 的版本也带有该后缀。运行时首次缺少图计算扩展时，也会补上对应 wheel 源。macOS 不支持 NVIDIA CUDA，作者仓库计划要求 CUDA 时会明确报错，不会静默改成 CPU 或 MPS。

Apple Silicon / Python 3.12 的 GPRGNN 依赖已通过真实安装与张量、稀疏矩阵、图卷积检查；这不等于论文训练完成，也不覆盖 Intel Mac 或任意依赖组合。旧的失败记录保留，新版本清单使用新的缓存摘要，无需删除整个依赖目录。

## 保留实现：健康检查与增量复用

以下机制已保留在依赖执行器中，供启用 `dependency_health_check` 的调用路径使用；当前作者仓库与固定预设入口未默认启用，也没有网页开关可开启它们。

1. 优先使用原清单的就绪环境，检查包版本、实际导入位置及原生运算。
2. 在同一 Python ABI、系统、架构与下载源策略下，查找包含相同依赖要求的已验证环境。满足要求即可直接复用，包括离线运行。
3. 需要新增包时，使用 pip 的 `--dry-run --ignore-installed --report` 解析完整 wheel 依赖闭包。按实际安装元数据比较版本，选择可复用的基础环境；只安装缺少的 wheel，每个 wheel 使用解析得到的 HTTPS 地址与 SHA-256。
4. 增量目录记录基础环境引用、完整解析版本、共享包与新增包。执行和独立评价按“增量目录 → 基础环境”导入；健康检查核对完整解析版本，拒绝以宿主全局包补足缺失依赖。

基础环境保持原样，增量安装不修改其中的包。不同版本可以在增量层覆盖，原有基础环境及使用它的其他任务保持各自的版本。增量组合无法通过原生健康检查时，保留原因并尝试独立安装原清单。该增量复用实现覆盖普通 wheel 清单与已支持的公开 PyTorch 补充索引；私有源、源码地址、extras、marker 和嵌套清单保持原独立安装行为。

## 空间与回收

当前可通过下面的 CLI 手动回收。安装前自动回收也已保留在执行器中，但只在启用上述健康检查的仓库依赖安装路径中触发。

| 设置 | 默认值 | 用途 |
| --- | --- | --- |
| `AUTOREPRO_DEPS_ROOT` | 数据根下的 `deps/` | 依赖缓存位置，可改到其他磁盘；数据根由 `AUTOREPRO_DATA_ROOT` 指定，默认是项目 `data/` |
| `AUTOREPRO_DEPS_QUOTA_GB` | 20 GiB | 手动回收及启用自动回收时的缓存容量目标 |
| `AUTOREPRO_DEPS_MIN_FREE_GB` | 2 GiB | 回收时的最低可用磁盘空间目标；自动安装路径启用时还会检查该余量 |

回收先处理失败或未完成的环境，再按最后使用时间处理闲置环境。安装和作者命令执行使用同一缓存锁；外部清理遇到活动任务直接跳过。增量环境仍引用的基础目录不能单独删除；同时选择增量与基础目录时先删除增量目录。超过 30 天未使用的环境也会进入回收候选。

损坏的最后使用时间回退到目录修改时间，基础环境引用统一按跨平台路径规则解析。遇到符号链接或读取失败而无法完整计量的环境会保留，同时保护其已知基础环境；回执通过 `errors` 和 `unmeasured_environments` 列出问题，此时 `remaining_bytes` 仅包含成功计量的目录，`quota_satisfied` 为 `false`。容量、剩余空间和保留天数必须为有限非负数，非法参数会在回收前报错。

容量目标只约束可回收缓存；仍被保留环境引用的基础包可能使缓存暂时超过目标。最低可用空间目标不能保证任意大小的未知依赖或训练产物都足够，也不代表当前默认安装入口会执行余量检查。PDF、真实数据、模型产物、运行日志和报告不在这项回收范围内。下载缓存仍由 pip 管理。

```powershell
python scripts/resource_cli.py deps-prune                       # 预览
python scripts/resource_cli.py deps-prune --yes                 # 执行
python scripts/resource_cli.py deps-prune --cache-gb 12 --min-free-gb 4 --yes
```

网页历史记录中的依赖表显示缓存目录占用，以及已有增量环境的基础引用。运行证据中的 `dependency_paths` 包含实际使用的全部依赖目录；只有触发相应保留机制时，`dependency_reuse` 才记录共享与新增包，`dependency_cache_maintenance` 才记录自动回收清单与磁盘检查结果。

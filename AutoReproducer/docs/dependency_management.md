# 依赖环境管理

此前按完整 requirements 文本创建隔离环境。即使只是多一个小包，也会重新安装一整套 PyTorch；不同 Python 版本和 CPU／CUDA 环境的缓存还会继续累积。2026-10-12 的 SGC 回归因此在环境准备阶段遇到 `No space left on device`，训练尚未启动。

## 安装与复用

1. 优先使用原清单的就绪环境，检查包版本、实际导入位置及原生运算。
2. 在同一 Python ABI、系统、架构与下载源策略下，查找包含相同依赖要求的已验证环境。满足要求即可直接复用，包括离线运行。
3. 需要新增包时，使用 pip 的 `--dry-run --ignore-installed --report` 解析完整 wheel 依赖闭包。按实际安装元数据比较版本，选择可复用的基础环境；只安装缺少的 wheel，每个 wheel 使用解析得到的 HTTPS 地址与 SHA-256。
4. 增量目录记录基础环境引用、完整解析版本、共享包与新增包。执行和独立评价按“增量目录 → 基础环境”导入；健康检查核对完整解析版本，拒绝以宿主全局包补足缺失依赖。

基础环境保持原样，增量安装不修改其中的包。不同版本可以在增量层覆盖，原有基础环境及使用它的其他任务保持各自的版本。增量组合无法通过原生健康检查时，保留原因并尝试独立安装原清单。当前自动增量复用覆盖普通 wheel 清单与已支持的公开 PyTorch 补充索引；私有源、源码地址、extras、marker 和嵌套清单保持原独立安装行为。

## 空间与回收

| 设置 | 默认值 | 用途 |
| --- | --- | --- |
| `AUTOREPRO_DEPS_ROOT` | 项目 `data/deps` | 依赖缓存位置，可改到其他磁盘 |
| `AUTOREPRO_DEPS_QUOTA_GB` | 20 GiB | 新安装前的缓存容量目标 |
| `AUTOREPRO_DEPS_MIN_FREE_GB` | 2 GiB | 安装前要求的最低可用磁盘空间 |

回收先处理失败或未完成的环境，再按最后使用时间处理闲置环境。安装和作者命令执行使用同一缓存锁；外部清理遇到活动任务直接跳过。增量环境仍引用的基础目录不能单独删除；同时选择增量与基础目录时先删除增量目录。超过 30 天未使用的环境也可自动回收。

容量目标只约束可回收缓存；仍被保留环境引用的基础包可能使缓存暂时超过目标。最低可用空间是安装前的余量检查，不能保证任意大小的未知依赖或训练产物都足够。PDF、真实数据、模型产物、运行日志和报告不在这项回收范围内。下载缓存仍由 pip 管理。

```powershell
python scripts/resource_cli.py deps-prune                       # 预览
python scripts/resource_cli.py deps-prune --yes                 # 执行
python scripts/resource_cli.py deps-prune --cache-gb 12 --min-free-gb 4 --yes
```

网页历史记录中的依赖表显示基础环境引用和增量目录实际占用。运行证据中的 `dependency_paths` 包含实际使用的全部依赖目录，`dependency_reuse` 说明共享与新增包，`dependency_cache_maintenance` 保存回收清单与磁盘检查结果。

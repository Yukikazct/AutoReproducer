# 文件与存储管理

`src/project_paths.py` 统一应用的数据目录。未配置时使用项目所在目录的 `data/`，不随启动时的工作目录改变。资源下载、审计日志、报告及图片、网页历史记录、进度文件和资源事件使用同一数据根。

## 配置目录

在启动应用前设置 `AUTOREPRO_DATA_ROOT`。`AUTOREPRO_DEPS_ROOT` 可以把共享依赖缓存放到另一块磁盘；未设置时使用数据根下的 `deps/`。路径支持 `~`，相对路径以调用者的当前工作目录为基准，建议使用绝对路径。

```bash
export AUTOREPRO_DATA_ROOT=/mnt/research/autoreproducer
export AUTOREPRO_DEPS_ROOT=/mnt/cache/autoreproducer-deps
python -m streamlit run app.py
```

Windows PowerShell 使用 `$env:AUTOREPRO_DATA_ROOT = "E:\AutoReproducer-data"` 和 `$env:AUTOREPRO_DEPS_ROOT = "E:\AutoReproducer-deps"`。

CLI 的 `--data-root` 放在子命令前，仅指定本次资源操作的数据目录。`resources` 和 `events` 同样支持该选项；独立设置的 `AUTOREPRO_DEPS_ROOT` 仍决定共享依赖清单的位置。

```bash
python scripts/resource_cli.py --data-root /mnt/research/autoreproducer status
python scripts/resource_cli.py --data-root /mnt/research/autoreproducer resources --json
python scripts/resource_cli.py --data-root /mnt/research/autoreproducer events --json
```

配置只改变后续读写位置，不会自动移动已有文件。整体迁移时，先停止运行任务，再复制数据目录；单篇论文的代码、数据集和本地权重可以用归档/恢复迁移，报告与运行记录需另行保留。

## 归档与恢复

```bash
python scripts/resource_cli.py --data-root /old/data archive PAPER_ID --dest /backup
python scripts/resource_cli.py --data-root /new/data restore /backup/PAPER_ID.zip
```

归档先写入同目录临时文件并校验 ZIP，再原子替换正式归档。恢复拒绝越界路径、链接、重复路径以及包含多篇论文的混合归档；先在暂存目录完整解压，CRC 或格式错误不会覆盖现有资源。空目录会保留，POSIX 系统还会恢复文件执行权限。恢复后的 manifest 指向当前数据根，并清除旧的 `cleaned_at` 标记。

恢复写入按文件替换；如果最终写入阶段发生磁盘或权限错误，返回失败和已经恢复的文件清单，可修复问题后重试。整个资源树不提供事务回滚。符号链接和特殊文件暂不支持归档，遇到时明确报错并保留原资源。

## 清理与统计

`prune` 默认只预览，`prune --yes` 先归档再清理。归档失败或清理不完整时返回错误，仍超配额时返回退出码 2。资源清理仅将实际删除成功的目录记入结果，失败资源继续保留在 manifest 中。

网页历史记录删除和目录统计跳过符号链接、Windows junction，避免操作外部目标；删除失败不计入已释放空间。依赖缓存清理使用缓存锁，并保护仍被增量环境引用的基础目录。资源事件按行读取，只在内存保留请求数量的末尾记录，跳过未写完或损坏的 JSON 行。

## Windows 验证

路径和归档规则在各系统上一致：拒绝 Windows 保留设备名、末尾空格/点、非法字符，以及大小写或 Unicode 规范化后的路径冲突，避免同一归档在 Windows 上覆盖不同条目。目录边界检查包括 Windows junction；缓存的基础环境引用同时识别 `/` 与 `\\`。

临时文件关闭后再替换目标文件，Windows 短暂共享冲突采用有限重试；持续被其他程序占用时返回错误并保留旧文件。清理 Git 对象及依赖缓存遇到普通只读文件时，增加写权限并重试删除；锁定、链接或其他权限错误仍会报告失败。POSIX 执行位不会在 Windows 上转成只读属性。托管子进程明确使用 UTF-8 输入输出，支持中文和带空格的工作目录。

GitHub Actions 覆盖 Windows/Linux 和 Python 3.11/3.12。先安装生产依赖并独立检查，再安装测试依赖；Windows 同时运行默认编码与 UTF-8 模式。Windows 专属用例使用真实 junction 和文件占用句柄，macOS/Linux 会明确跳过这些用例。修改 CI 配置本身不代表 Windows 验收已经通过，需以 Windows runner 产生的测试报告为准。

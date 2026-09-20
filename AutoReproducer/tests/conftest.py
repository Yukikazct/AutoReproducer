"""测试级隔离配置。

code_executor 在模块导入期读取 AUTOREPRO_DEPS_ROOT 计算 DEPS_CACHE_ROOT；
若不在收集阶段提前设置，单测中 mock 的"pip 安装成功"会真实写入项目
data/deps/（留下只有 .ready 标记的空目录），污染 L0 依赖缓存。
conftest 在 pytest 收集任何被测模块之前加载，这里统一把依赖缓存根
指向一次性临时目录，并保证仓库根可导入 src。
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "AUTOREPRO_DEPS_ROOT",
    str(Path(tempfile.mkdtemp(prefix="autorepro_deps_test_"))),
)
# ResourceManager 的数据根同样指向一次性临时目录，
# 防止 e2e 流水线把 manifest/数据集写入项目 data/ 造成污染。
os.environ.setdefault(
    "AUTOREPRO_DATA_ROOT",
    str(Path(tempfile.mkdtemp(prefix="autorepro_data_test_"))),
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
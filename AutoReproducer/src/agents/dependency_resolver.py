"""dependency_resolver - 静态依赖解析工具（P1-⑧，融合 ScholarAgent coder.py）。

职责（纯函数、无 LLM、无网络副作用）：
1. detect_code_dependencies:   AST 解析源码 import / from ... import -> 第三方顶
   层模块名（过滤标准库、相对导入、语法错误安全返回空）;
2. detect_repo_dependencies:   扫描仓库 requirements.txt / environment.yml /
   pyproject.toml -> 依赖清单（限深、跳过噪音目录）;
3. filter_standard_library:    requirements token 级 stdlib 过滤（防"os==.."
   这类假包混进安装清单）;
4. normalize_py39:             已知 Python 3.9 不兼容 pin 的确定性归一代换;
5. find_missing_module / python_package_for:
   从运行 stderr 提取缺失模块名并映射 PyPI 包名（运行时 pip 自愈用）。

与 ScholarAgent 的关系: 其正则 import 解析升级为 ast 遍历（更准、天然支持
try/except 内 import 与别名）；requirements 解析收敛为逐行 token 化并去
环境标记（`pkg==1.0; python_version<"3.10"` -> 只留 `pkg==1.0`）。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

# ---------------------------------------------------------------- 标准库

# Python 3.10+ 自带 stdlib 模块名集合；3.9 兜底补常用量。
_STDLIB_CORE = {
    "abc", "argparse", "array", "ast", "asyncio", "base64", "bisect", "builtins",
    "bz2", "calendar", "cmath", "collections", "colorsys", "concurrent",
    "contextlib", "copy", "csv", "ctypes", "dataclasses", "datetime", "decimal",
    "difflib", "dis", "email", "enum", "errno", "faulthandler", "fractions",
    "functools", "gc", "getopt", "getpass", "glob", "graphlib", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "importlib", "inspect", "io", "ipaddress",
    "itertools", "json", "keyword", "linecache", "locale", "logging", "lzma",
    "math", "mimetypes", "multiprocessing", "numbers", "operator", "os",
    "pathlib", "pickle", "platform", "plistlib", "pprint", "profile", "pstats",
    "queue", "random", "re", "readline", "reprlib", "sched", "secrets",
    "select", "selectors", "shelve", "shlex", "shutil", "signal", "site",
    "socket", "sqlite3", "ssl", "stat", "statistics", "string", "struct",
    "subprocess", "sys", "sysconfig", "tarfile", "tempfile", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize", "tomllib",
    "trace", "traceback", "types", "typing", "unicodedata", "unittest", "urllib",
    "uuid", "venv", "warnings", "wave", "weakref", "webbrowser", "xml", "xmlrpc",
    "zipfile", "zipimport", "zlib",
}
STDLIB_MODULES: Set[str] = (
    set(getattr(sys, "stdlib_module_names", ())) | _STDLIB_CORE)

# ---------------------------------------------------------------- PyPI 映射

# import 模块名 -> PyPI 包名（名称不一致的常见包）
MODULE_TO_PYPI = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "huggingface_hub": "huggingface-hub",
    "tensorflow_hub": "tensorflow-hub",
    "tensorflow_datasets": "tensorflow-datasets",
    "tensorboardX": "tensorboardX",
    "google.protobuf": "protobuf",
    "_tkinter": "tk",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "torch": "torch",
    "transformers": "transformers",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "requests": "requests",
    "flask": "flask",
    "beautifulsoup": "beautifulsoup4",
    "nltk": "nltk",
    "pytest": "pytest",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "sentencepiece": "sentencepiece",
}
# PyPI 包名校验：requirements/自愈安装只接受合法包名
_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def python_package_for(module: str) -> str:
    """模块名 -> PyPI 安装名（未知时保真返回模块名）。"""
    module = (module or "").split(".")[0]
    return MODULE_TO_PYPI.get(module, module)


def dependency_root(token: str) -> str:
    """requirements token 的包根名：剥离版本/标记/空白并小写。

    "numpy>=1.26,<3; python_version<'3.10'" -> "numpy"
    """
    token = (token or "").strip()
    token = token.split(";")[0].split("#")[0].strip()   # 去环境标记与行内注释
    root = re.split(r"[<>=!~\[(]", token, maxsplit=1)[0].strip()
    return root.lower()


# ---------------------------------------------------------------- 代码解析

def dedupe(dependencies: Iterable[str]) -> List[str]:
    """保序去重（大小写不敏感，保留首现形式）。"""
    seen: Set[str] = set()
    out: List[str] = []
    for dep in dependencies:
        dep = (dep or "").strip()
        if not dep:
            continue
        key = dep.lower()
        if key not in seen:
            seen.add(key)
            out.append(dep)
    return out


def detect_code_dependencies(code: str) -> List[str]:
    """AST 解析源码 import 语句 -> 第三方顶级模块名列表（保序去重）。

    - 跳过标准库模块与相对导入（from . import x / from ..pkg import y）;
    - try/except 内的 import（常见于兼容导入）同样被 ast.walk 捕获;
    - 语法错误时返回空（不抛异常——依赖解析不应阻断流水线）。
    """
    if not code or not code.strip():
        return []
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    modules: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".")[0]
                if root and root not in STDLIB_MODULES:
                    modules.append(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # 相对导入：跳过
                continue
            root = (node.module or "").split(".")[0]
            if root and root not in STDLIB_MODULES:
                modules.append(root)
    return dedupe(modules)


# ---------------------------------------------------------------- 仓库解析

_SKIP_REPO_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "site-packages",
    "__pycache__", ".cache", "dist", "build", "data", "datasets", "checkpoints",
    "wandb", ".tox", ".nox", ".idea", ".vscode", "results", "outputs", "logs",
}


def _iter_repo_files(repo_path: str, max_depth: int = 3):
    """限深遍历仓库文件，跳过噪音目录与超大文件。"""
    root = Path(repo_path)
    if not root.is_dir():
        return
    try:
        top_depth = len(root.resolve().parts)
    except OSError:
        return
    for current in Path(root).rglob("*"):
        if current.is_dir():
            continue
        rel = current.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        if any(part in _SKIP_REPO_DIRS for part in rel.parts[:-1]):
            continue
        try:
            if current.stat().st_size > 2 * 1024 * 1024:   # >2MB 跳过
                continue
        except OSError:
            continue
        yield current


def _parse_requirements_line(line: str) -> Optional[str]:
    """解析 requirements.txt 单行：返回纯依赖 token 或 None。"""
    line = (line or "").strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith(("-r ", "-c ", "-e ", "--", "-f ", "-i ")):
        return None
    token = line.split("#", 1)[0].split(";", 1)[0].strip()
    if not token:
        return None
    return token


def _parse_environment_yml(text: str) -> List[str]:
    """文本级解析 environment.yml 的 dependencies 段（避免硬依赖 yaml）。

    - conda 项（`- numpy=1.26`）归一为 pip 合法版本符（`numpy==1.26`）；
    - `- pip:` 子列表（`- pandas==2.1`）原样保留；
    - python / pip 本身不进入依赖清单。
    """
    found: List[str] = []
    inside = False
    in_pip = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line.startswith("dependencies:"):
            inside, in_pip = True, False
            continue
        if not inside:
            continue
        if line.startswith("- pip:"):
            in_pip = True
            continue
        if line.startswith("- ") and not in_pip:
            token = line[2:].strip()
            name = token.split("=")[0].strip().lower()
            if name in {"python", "pip", ""}:
                continue
            # conda 版本符 `=` -> pip `==`
            if "=" in token and "==" not in token and "<" not in token \
                    and ">" not in token:
                token = token.replace("=", "==", 1)
            found.append(token)
        elif line.startswith("- "):        # pip 子列表项，原样保留
            token = line[2:].strip()
            if token and not token.startswith("-"):
                found.append(token)
        elif line and not line.startswith("-"):
            break                           # 离开 dependencies 段
    return found


def _parse_pyproject_dependencies(text: str) -> List[str]:
    """解析 pyproject.toml 的 [project] dependencies（tomllib，3.11+）。"""
    try:
        import tomllib
        data = tomllib.loads(text)
    except Exception:
        return []
    found: List[str] = []
    for dep in data.get("project", {}).get("dependencies", []) or []:
        dep = str(dep).split(";")[0].strip()
        if dep:
            found.append(dep)
    return found


def detect_repo_dependencies(repo_path: str, max_depth: int = 3) -> List[str]:
    """扫描仓库的 requirements/environment/pyproject -> 依赖清单（保序去重）。"""
    root = Path(repo_path)
    if not root.is_dir():
        return []
    found: List[str] = []
    for path in _iter_repo_files(repo_path, max_depth=max_depth):
        name = path.name.lower()
        try:
            if name == "requirements.txt":
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    token = _parse_requirements_line(line)
                    if token:
                        found.append(token)
            elif name in ("environment.yml", "environment.yaml"):
                found.extend(_parse_environment_yml(
                    path.read_text(encoding="utf-8", errors="replace")))
            elif name == "pyproject.toml":
                found.extend(_parse_pyproject_dependencies(
                    path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return dedupe(found)


def resolve_dependencies(code: Optional[str] = None,
                         repo_path: Optional[str] = None) -> List[str]:
    """双来源静态依赖解析（代码 import + 仓库声明文件）。

    合并后统一：去重 -> stdlib 过滤 -> py39 归一。
    代码 import 的模块名先经 MODULE_TO_PYPI 映射为 PyPI 包名
    （cv2->opencv-python、sklearn->scikit-learn 等），保证合并进
    requirements 的是可安装的包名而非导入名。
    """
    found: List[str] = []
    if code:
        found.extend(
            python_package_for(module)
            for module in detect_code_dependencies(code))
    if repo_path:
        found.extend(detect_repo_dependencies(repo_path))
    found = dedupe(found)
    found = filter_standard_library(found)
    found = normalize_py39(found)
    return found


# ---------------------------------------------------------------- 过滤与归一

def filter_standard_library(dependencies: Iterable[str]) -> List[str]:
    """按包根名过滤标准库依赖（requirements token 级别）。"""
    out: List[str] = []
    for dep in dependencies:
        root = dependency_root(dep)
        # 反向映射：stdlib 以别名/包名出现也应过滤（如 pip 包 "os"）
        if not root or root in STDLIB_MODULES:
            continue
        out.append(dep)
    return out


def normalize_py39(dependencies: Iterable[str]) -> List[str]:
    """已知 Python 3.9 不兼容 pin 的确定性归一代换（对齐 ScholarAgent）。"""
    normalized: List[str] = []
    for dep in dependencies:
        name = dependency_root(dep)
        pin = dep[len(name):] if dep.lower().startswith(name) else ""
        if name == "oscar":
            normalized.append("django-oscar==2.2")
        elif name == "pytorch":
            normalized.append("torch" + (pin if pin else ""))
        elif name == "tensorflow" and pin.startswith("==1.1"):
            continue                    # TF 1.1x 在 py3.9 无可用 wheel，直接剔除
        elif name == "llama-index":
            normalized.append("llama-index<0.12")
            if not any(dependency_root(d) == "pydantic" for d in normalized):
                normalized.append("pydantic<2.10")
        elif name == "langchain":
            normalized.append(dep)
            if not any(dependency_root(d) == "langchain-community"
                       for d in normalized):
                normalized.append("langchain-community")
        else:
            normalized.append(dep)
    return dedupe(normalized)


# ---------------------------------------------------------------- 缺失模块

_MISSING_MODULE_RES = (
    re.compile(r"ModuleNotFoundError:\s*No module named ['\"]([A-Za-z0-9_.]+)['\"]"),
    re.compile(r"ImportError:\s*No module named ['\"]([A-Za-z0-9_.]+)['\"]"),
    re.compile(r"ImportError:\s*cannot import name ['\"]([A-Za-z0-9_.]+)['\"] "
               r"from ['\"]([A-Za-z0-9_.]+)['\"]"),
    re.compile(r"ModuleNotFoundError:\s*No module named\s+([A-Za-z0-9_.]+)"),
)


def find_missing_module(stderr: str) -> Optional[str]:
    """从运行错误输出提取缺失的顶层模块名；未命中返回 None。

    命中规则：ModuleNotFoundError / ImportError 的常见文本形态；
    `cannot import name X from Y` 场景取 Y（真正要装的包）。
    """
    if not stderr:
        return None
    for pattern in _MISSING_MODULE_RES:
        match = pattern.search(stderr)
        if not match:
            continue
        try:
            module = match.group(2) if match.lastindex and match.lastindex >= 2 \
                else match.group(1)
        except IndexError:                     # 单捕获组表达式
            module = match.group(1)
        module = (module or "").split(".")[0]
        if module and _PKG_NAME_RE.match(module):
            return module
    return None
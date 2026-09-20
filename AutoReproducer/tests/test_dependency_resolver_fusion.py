"""dependency_resolver 静态依赖解析测试（P1-⑧）。

覆盖：
1. detect_code_dependencies：AST 提取 import/from、过滤 stdlib 与相对导入、
   语法错误安全返回空；
2. detect_repo_dependencies：requirements.txt / environment.yml /
   pyproject.toml 三来源解析，跳过 .git 等噪音目录与超大文件；
3. filter_standard_library：requirements token 级 stdlib 过滤；
4. normalize_py39：已知 py39 不兼容 pin 的归一代换；
5. find_missing_module / python_package_for：缺失模块识别与 PyPI 映射；
6. resolve_dependencies 双来源融合：模块名 -> PyPI 包名、去重、归一；
7. EnvBuilderAgent 融合：静态解析补漏不覆盖，env_config 带 static 字段。

运行: python -m pytest tests/test_dependency_resolver_fusion.py -v
"""
import sys
from pathlib import Path

import pytest

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.dependency_resolver import (  # noqa: E402
    detect_code_dependencies, detect_repo_dependencies,
    dependency_root, filter_standard_library, find_missing_module,
    normalize_py39, python_package_for, resolve_dependencies,
)
from src.agents.env_builder import EnvBuilderAgent  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402


# ---------------- detect_code_dependencies ----------------

def test_detect_basic_imports():
    deps = detect_code_dependencies(
        "import numpy as np\nimport cv2\nimport os\nimport sys\n"
        "from sklearn.svm import SVC\n"
        "import torch.nn as nn\n")
    assert deps == ["numpy", "cv2", "sklearn", "torch"]


def test_detect_skips_stdlib_and_relative():
    deps = detect_code_dependencies(
        "import math\nimport json\nfrom os.path import join\n"
        "from . import helper\nfrom ..pkg import mod\n"
        "import argparse\n")
    assert deps == []


def test_detect_try_except_imports():
    deps = detect_code_dependencies(
        "try:\n    import torch\n"
        "except ImportError:\n    import numpy as np\n")
    assert deps == ["torch", "numpy"]


def test_detect_syntax_error_safe():
    assert detect_code_dependencies("def broken(:\n  pass") == []
    assert detect_code_dependencies("") == []
    assert detect_code_dependencies(None) == []  # type: ignore[arg-type]


def test_detect_dedupe_preserve_order():
    deps = detect_code_dependencies(
        "import numpy\nimport pandas\nimport numpy as np\n")
    assert deps == ["numpy", "pandas"]


# ---------------- detect_repo_dependencies ----------------

def _write(tmp_path, name, content):
    f = tmp_path / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f


def test_repo_requirements_txt(tmp_path):
    _write(tmp_path, "requirements.txt",
           "# comment\nnumpy>=1.26,<3\n\n"
           "matplotlib==3.8; python_version<'3.11'\n-r other.txt\n")
    deps = detect_repo_dependencies(str(tmp_path))
    assert deps == ["numpy>=1.26,<3", "matplotlib==3.8"]


def test_repo_environment_yml(tmp_path):
    _write(tmp_path, "environment.yml",
           "name: demo\nchannels:\n  - conda-forge\n"
           "dependencies:\n  - python=3.11\n  - numpy=1.26\n"
           "  - pip:\n    - pandas==2.1\n    - scikit-learn==1.3\n")
    deps = detect_repo_dependencies(str(tmp_path))
    assert sorted(deps) == ["numpy==1.26", "pandas==2.1", "scikit-learn==1.3"]


def test_repo_pyproject_toml(tmp_path):
    _write(tmp_path, "pyproject.toml",
           "[project]\nname = \"demo\"\nversion = \"0.1\"\n"
           "dependencies = [\"numpy>=1.24\", \"torch==2.1.2\"]\n")
    deps = detect_repo_dependencies(str(tmp_path))
    assert deps == ["numpy>=1.24", "torch==2.1.2"]


def test_repo_skips_noise_dirs_and_big_files(tmp_path):
    _write(tmp_path, "requirements.txt", "numpy>=1.24")
    _write(tmp_path, ".git/requirements.txt", "evil_pkg")
    _write(tmp_path, "node_modules/requirements.txt", "node_pkg")
    big = tmp_path / "env/data/requirements.txt"
    big.parent.mkdir(parents=True, exist_ok=True)
    big.write_text("big_pkg")
    deps = detect_repo_dependencies(str(tmp_path))
    assert deps == ["numpy>=1.24"]


def test_repo_missing_dir_empty():
    assert detect_repo_dependencies("/nonexistent/path/xyz") == []


# ---------------- filter_standard_library ----------------

def test_filter_stdlib_tokens():
    assert filter_standard_library(
        ["numpy>=1.24", "os", "sys==3.0", "json; python_version<'3.11'"]) \
        == ["numpy>=1.24"]


# ---------------- normalize_py39 ----------------

def test_normalize_py39_known_rules():
    deps = normalize_py39(["numpy>=1.24", "oscar==2.1", "pytorch",
                           "tensorflow==1.15", "llama-index==0.10",
                           "langchain==0.1"])
    names = [dependency_root(d) for d in deps]
    assert "numpy>=1.24" in deps
    assert "django-oscar==2.2" in deps          # oscar -> django-oscar pin
    assert "torch" in deps and "pytorch" not in deps
    assert "tensorflow" not in deps             # TF 1.1x 剔除
    assert "llama-index<0.12" in deps
    assert any(n.startswith("pydantic") for n in names)
    assert "langchain==0.1" in deps
    assert "langchain-community" in deps


# ---------------- find_missing_module / python_package_for ----------------

@pytest.mark.parametrize("stderr,expected", [
    ("ModuleNotFoundError: No module named 'cv2'", "cv2"),
    ("ImportError: No module named 'sklearn'", "sklearn"),
    ("Traceback...\nModuleNotFoundError: No module named 'torch.nn'"
     .replace("\n", "\n"), "torch"),
    ("from PIL import Image" + " -> ImportError: cannot import name 'Image' "
     "from 'PIL'", "PIL"),
])
def test_find_missing_module(stderr, expected):
    assert find_missing_module(stderr) == expected


def test_find_missing_module_none():
    assert find_missing_module("") is None
    assert find_missing_module("ValueError: bad input") is None


def test_python_package_for_mapping():
    assert python_package_for("cv2") == "opencv-python"
    assert python_package_for("sklearn") == "scikit-learn"
    assert python_package_for("PIL") == "pillow"
    assert python_package_for("numpy.sub") == "numpy"
    assert python_package_for("unknown_pkg") == "unknown_pkg"


# ---------------- resolve_dependencies 双来源融合 ----------------

def test_resolve_code_imports_mapped_to_pypi():
    deps = resolve_dependencies(
        code="import cv2\nimport os\nimport numpy as np\nfrom sklearn import svm\n")
    # cv2 -> opencv-python, sklearn -> scikit-learn, stdlib 过滤; numpy 保真
    assert "opencv-python" in deps
    assert "scikit-learn" in deps
    assert "numpy" in deps
    assert "os" not in deps


def test_resolve_code_and_repo_fusion(tmp_path):
    _write(tmp_path, "requirements.txt", "numpy>=1.24")
    deps = resolve_dependencies(code="import pandas\nimport cv2",
                                repo_path=str(tmp_path))
    assert deps == ["pandas", "opencv-python", "numpy>=1.24"]


# ---------------- EnvBuilderAgent 融合 ----------------

def _env_agent():
    return EnvBuilderAgent(LLMClient(mock_mode=True))


def test_env_builder_static_deps_complement(tmp_path):
    """静态解析补充 LLM/声明依赖缺失的包，且不覆盖声明版本 pin。"""
    agent = _env_agent()
    input_data = {
        "paper_info": {"method": "分类", "dependencies": ["numpy>=1.24"]},
        "resources": {},
        "corpus_paper": str(tmp_path / "missing.txt"),  # 无语料，走 LLM
        "code": "import numpy as np\nimport cv2\nfrom sklearn import svm\n"
                "import matplotlib.pyplot as plt\n",
    }
    res = agent.run(input_data)
    env = res["env_config"]
    reqs = env["requirements_txt"].splitlines()

    assert env["static_source"] == "code"
    static = env["static_dependencies"]
    assert "opencv-python" in static          # cv2 映射
    assert "scikit-learn" in static           # sklearn 映射
    # LLM 兜底 torch>=2.0 与声明 numpy 均在，静态补的包不覆盖版本
    assert any(r.startswith("numpy") for r in reqs)  # 声明版本不覆盖
    assert any(r.startswith("torch") for r in reqs)
    # 静态解析补充后 requirements 应含 opencv-python / scikit-learn
    assert any(r.lower().startswith("opencv") for r in reqs)
    assert any(r.lower().startswith("scikit-learn") for r in reqs)


def test_env_builder_no_code_no_static(tmp_path):
    """无代码输入时 static 字段为空，行为与改造前一致。"""
    agent = _env_agent()
    input_data = {
        "paper_info": {"method": "分类", "dependencies": ["numpy>=1.24"]},
        "resources": {},
        "corpus_paper": str(tmp_path / "missing.txt"),
    }
    res = agent.run(input_data)
    env = res["env_config"]
    assert env["static_dependencies"] == []
    assert env["static_source"] == "none"


def test_env_builder_corpus_deps_authoritative(tmp_path, monkeypatch):
    """语料 requirements 权威：静态解析只补 src 缺失，不覆盖语料版本。"""
    monkeypatch.setattr(
        EnvBuilderAgent, "_load_corpus_deps",
        staticmethod(lambda paper: ["torch==2.0.1", "numpy==1.24.0"]))
    agent = _env_agent()
    input_data = {
        "paper_info": {"method": "训练", "dependencies": ["torch>=2.0"]},
        "resources": {},
        "corpus_paper": "paper-xyz",  # 任意 id，由 monkeypatch 接管
        "code": "import torch\nimport pandas as pd\n",
    }
    res = agent.run(input_data)
    env = res["env_config"]
    reqs = env["requirements_txt"].splitlines()
    assert "torch==2.0.1" in reqs          # 语料 pin 保持
    assert "numpy==1.24.0" in reqs
    assert any(r.lower() == "pandas" for r in reqs)   # 静态补充
    assert env["static_source"] == "code"
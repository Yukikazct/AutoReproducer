"""多代码单元资源层测试（CodeUnit + ResourceManager.fetch_units）。

覆盖：
1. sanitize_unit_id：路径穿越防护（.. / 非法字符 / 空串）；
2. units_from_discovery：selected -> main、可信候选 -> alt_*、URL 去重；
3. fetch_units：多单元落盘 data/repos/<paper_id>/<unit_id>/，
   逐单元溯源标记、占位 URL 跳过、空 URL 跳过、单元失败不阻断其余；
4. build_manifest：多单元布局产出 code_units 字段，旧单仓库布局兼容；
5. save_plan/load_plan 与非法 paper_id 拒绝；
6. archive/restore 携带 plans 成员往返。

运行: python -m pytest tests/test_code_units.py -v
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.code_units import (  # noqa: E402
    CodeUnit,
    dedupe_units,
    sanitize_unit_id,
    units_from_discovery,
)
from src.resource_manager import ResourceManager  # noqa: E402


# ---------------- 夹具：本地 file:// git 仓库 ----------------

def _make_git_repo(root: Path, name: str = "repo", files: dict = None) -> str:
    """在 root/name 下 git init + commit，返回 file:// URL。"""
    repo_dir = root / name
    repo_dir.mkdir(parents=True)
    (repo_dir / "README.md").write_text("# test repo\n", encoding="utf-8")
    for rel, content in (files or {}).items():
        f = repo_dir / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email",
                    "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name",
                    "tester"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-q", "-m", "init"],
                   check=True)
    return f"file://{repo_dir}"


@pytest.fixture()
def rm(tmp_path):
    return ResourceManager(data_root=str(tmp_path / "data"))


# ---------------- 1. sanitize_unit_id ----------------

def test_sanitize_unit_id_rejects_traversal():
    assert sanitize_unit_id("..") == ""
    assert sanitize_unit_id(".") == ""
    assert sanitize_unit_id("") == ""
    assert sanitize_unit_id("../etc") == "_etc" or \
        sanitize_unit_id("../etc") == ".._etc"


def test_sanitize_unit_id_keeps_safe_chars():
    assert sanitize_unit_id("main") == "main"
    assert sanitize_unit_id("lib_0") == "lib_0"
    assert sanitize_unit_id("my-repo.v1") == "my-repo.v1"
    assert sanitize_unit_id("a/b") == "a_b"


# ---------------- 2. units_from_discovery ----------------

def test_units_from_discovery_maps_selected_to_main():
    discovery = {
        "selected_repo": "https://github.com/thuml/iTransformer",
        "discovery_chain": ["curated_fallback"],
        "candidates": [
            {"repo_urls": ["https://github.com/thuml/Time-Series-Library"],
             "role": "library", "source": "curated_fallback"},
        ],
    }
    units = units_from_discovery(discovery)
    assert units[0].unit_id == "main"
    assert units[0].url == "https://github.com/thuml/iTransformer"
    assert units[1].unit_id == "alt_0"
    assert units[1].url == "https://github.com/thuml/Time-Series-Library"


def test_dedupe_units_by_url():
    units = [
        CodeUnit(unit_id="main", url="https://github.com/a/b"),
        CodeUnit(unit_id="alt_1", url="https://github.com/a/b"),
        CodeUnit(unit_id="alt_2", url=""),
    ]
    out = dedupe_units(units)
    assert [u.unit_id for u in out] == ["main"]


# ---------------- 3. fetch_units 多单元落盘 ----------------

def test_fetch_units_clones_each_unit_with_provenance(rm, tmp_path):
    main_url = _make_git_repo(tmp_path, "main_repo",
                              {"run.py": "print('ok')\n"})
    lib_url = _make_git_repo(tmp_path, "lib_repo")

    infos = rm.fetch_units("pid1", [
        {"unit_id": "main", "role": "main", "url": main_url},
        {"unit_id": "lib_0", "role": "library", "url": lib_url},
        {"unit_id": "bad/../x", "role": "alt", "url": main_url},  # 非法 id
        {"unit_id": "skip", "role": "alt", "url": ""},            # 空 URL
        {"unit_id": "ph", "role": "alt",
         "url": "https://github.com/example/repo"},               # 占位
    ])
    assert len(infos) == 5
    by_id = {i["unit_id"]: i for i in infos}
    assert by_id["main"]["state"] == "cloned"
    assert by_id["lib_0"]["state"] == "cloned"
    assert by_id["skip"]["state"] == "skipped" or by_id["skip"]["detail"]
    assert by_id["ph"]["state"] == "placeholder-skip"

    main_dir = rm.repos_root / "pid1" / "main"
    assert main_dir.is_dir()
    assert (main_dir / "run.py").is_file()
    marker = json.loads(
        (main_dir / ".autorepro-repo-source.json").read_text(
            encoding="utf-8"))
    assert marker["repo_url"] == main_url
    assert marker["commit"]  # 有 HEAD sha
    # lib 单元与 main 隔离落盘
    assert (rm.repos_root / "pid1" / "lib_0" / "README.md").is_file()


def test_fetch_units_cached_reuse(rm, tmp_path):
    url = _make_git_repo(tmp_path, "cached_repo")
    first = rm.fetch_units("pid2", [
        {"unit_id": "main", "role": "main", "url": url}])
    assert first[0]["state"] == "cloned"
    second = rm.fetch_units("pid2", [
        {"unit_id": "main", "role": "main", "url": url}])
    assert second[0]["state"] == "cached"


# ---------------- 4. manifest code_units ----------------

def test_manifest_lists_code_units(rm, tmp_path):
    url = _make_git_repo(tmp_path, "manifest_repo")
    rm.fetch_units("pid3", [
        {"unit_id": "main", "role": "main", "url": url},
        {"unit_id": "lib_0", "role": "library",
         "url": "https://github.com/example/lib"},
    ])
    manifest = rm.build_manifest("pid3", paper_title="测试论文")
    assert "code_units" in manifest
    units = {u["unit_id"]: u for u in manifest["code_units"]}
    assert units["main"]["url"] == url
    assert units["main"]["commit"]
    assert manifest["resources"]["code"] == str(
        rm.repos_root / "pid3" / "main")


def test_manifest_legacy_single_repo_layout(rm, tmp_path):
    """旧布局（仓库文件直接在 repos/<paper_id>/ 根下）不受影响。"""
    repo_dir = rm.repos_root / "pid4"
    repo_dir.mkdir(parents=True)
    (repo_dir / "run.py").write_text("print(1)\n")
    manifest = rm.build_manifest("pid4")
    assert "code_units" not in manifest
    assert manifest["resources"]["code"] == str(repo_dir)


# ---------------- 5. save_plan / load_plan ----------------

def test_save_and_load_plan(rm):
    plan = {"paper_id": "pid5", "source": "llm",
            "steps": [{"step_id": "run_0", "kind": "run",
                       "cmd": "python run.py"}]}
    path = rm.save_plan(plan)
    assert Path(path).is_file()
    loaded = rm.load_plan("pid5")
    assert loaded["source"] == "llm"
    assert loaded["steps"][0]["step_id"] == "run_0"
    assert rm.load_plan("no_such_pid") is None


def test_save_plan_rejects_illegal_paper_id(rm):
    with pytest.raises(ValueError):
        rm.save_plan({"paper_id": "../etc"})
    with pytest.raises(ValueError):
        rm.save_plan({"paper_id": ""})


# ---------------- 6. archive/restore 携带 plans ----------------

def test_archive_restore_roundtrip_with_plans(rm, tmp_path):
    url = _make_git_repo(tmp_path, "archive_repo")
    rm.fetch_units("pid6", [{"unit_id": "main", "role": "main", "url": url}])
    rm.save_plan({"paper_id": "pid6", "source": "heuristic",
                  "steps": []})

    archived = rm.archive("pid6")
    assert Path(archived["archive"]).is_file()
    assert archived["entries"] >= 2

    rm2 = ResourceManager(data_root=str(tmp_path / "data2"))
    restored = rm2.restore(archived["archive"], paper_id="pid6")
    assert restored["ok"] is True
    assert (rm2.repos_root / "pid6" / "main" / "README.md").is_file()
    loaded = rm2.load_plan("pid6")
    assert loaded is not None
    assert loaded["source"] == "heuristic"

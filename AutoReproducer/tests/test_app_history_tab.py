"""历史记录 Tab 的端到端测试（AppTest 驱动真实 app.py）。

用 Streamlit 的 AppTest 真的把 app.py 跑起来、真的点按钮，覆盖批量删除
最关键的两条性质——**只删可见且已勾选的会话**、**必须勾选确认才能删**——
以及「删除后的提示能跨 rerun 活下来」。

为什么必须走 AppTest：批量删除的坑几乎全在 Streamlit 的运行时语义里
（widget 选中值何时写入 session_state、什么时候能改 session_state、
st.rerun() 会不会丢掉提示），纯函数单测一条都覆盖不到。

数据目录经 monkeypatch 指向 tmp_path，不触碰真实 data/。

运行: python -m pytest tests/test_app_history_tab.py -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import frontend.history_manager as hm  # noqa: E402

APP_PATH = str(Path(__file__).parent.parent / "app.py")

SID_RUNNING = "20260910_100000"
SID_DONE = "20260911_000000"


def _mk_session(data_dir: Path, sid: str, title: str, state: str) -> None:
    """造一个最小可被 list_sessions 识别的会话（账本 + 日志 + 报告）。"""
    records = [{
        "type": "start",
        "outputs": {"title": title, "paper_info": {"title": title}},
        "result": {"state": state, "duration_sec": 1.0, "llm_calls": 2},
    }]
    ledger = data_dir / "experiment_ledger" / f"ledger_{sid}.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    log = data_dir / "logs" / f"session_{sid}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"type": "log"}) + "\n", encoding="utf-8")

    report = data_dir / "reports" / f"{title}_{sid}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# 报告", encoding="utf-8")


def _session_files(data_dir: Path, sid: str) -> list:
    return [data_dir / "experiment_ledger" / f"ledger_{sid}.jsonl",
            data_dir / "logs" / f"session_{sid}.jsonl"]


@pytest.fixture
def at(tmp_path, monkeypatch):
    """把历史数据目录指向 tmp_path、造两个会话，返回已渲染的 AppTest。"""
    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    _mk_session(tmp_path, SID_RUNNING, "Dummy Paper", "RUNNING")
    _mk_session(tmp_path, SID_DONE, "ResNet 复现", "COMPLETED")

    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()
    return app


def _captions(app) -> str:
    return " | ".join(c.value for c in app.caption)


def test_history_tab_has_per_session_checkbox(at):
    """每个会话都要有一个展开区之外的勾选框（批量删除的前提）。"""
    assert not at.exception
    keys = [c.key for c in at.checkbox]
    assert f"hist_chk_{SID_RUNNING}" in keys
    assert f"hist_chk_{SID_DONE}" in keys
    assert "confirm_batch_del" in keys


def test_filter_narrows_visible_list(at):
    """状态筛选后「可见」条数随之变化，并提示可见列表里有 RUNNING。"""
    at.selectbox(key="hist_state_filter").set_value("RUNNING")
    at.run()
    assert not at.exception
    assert "共 2 条 · 可见 1 条" in _captions(at)
    assert "RUNNING 会话" in _captions(at)


def test_batch_delete_requires_confirmation(at, tmp_path):
    """没勾确认框时批量删除按钮必须不可点。"""
    at.checkbox(key=f"hist_chk_{SID_DONE}").set_value(True)
    at.run()
    assert at.button(key="batch_del_btn").disabled is True

    at.checkbox(key="confirm_batch_del").set_value(True)
    at.run()
    assert at.button(key="batch_del_btn").disabled is False
    # 仅渲染、不点击：文件必须原封不动
    assert all(f.exists() for f in _session_files(tmp_path, SID_DONE))


def test_batch_delete_only_removes_visible_selected(at, tmp_path):
    """核心安全性质：勾选后又被筛选隐藏的会话，不得被删除。

    两条都勾上 -> 筛选到 RUNNING（只剩一条可见）-> 确认 -> 删除：
    只有可见的那条被删，被隐藏的 COMPLETED 必须完整保留。
    """
    at.checkbox(key=f"hist_chk_{SID_RUNNING}").set_value(True)
    at.checkbox(key=f"hist_chk_{SID_DONE}").set_value(True)
    at.run()
    assert "已选 2 条" in _captions(at)

    at.selectbox(key="hist_state_filter").set_value("RUNNING")
    at.run()
    assert "已选 1 条" in _captions(at)      # 隐藏的那条不计入

    at.checkbox(key="confirm_batch_del").set_value(True)
    at.run()
    assert "删除所选 1 条" in at.button(key="batch_del_btn").label
    at.button(key="batch_del_btn").click()
    at.run()
    assert not at.exception

    # RUNNING 那条（可见且已勾选）被删干净
    for f in _session_files(tmp_path, SID_RUNNING):
        assert not f.exists()
    assert not (tmp_path / "reports" / f"Dummy Paper_{SID_RUNNING}.md").exists()
    # COMPLETED 那条虽被勾选、但被筛选隐藏 -> 必须原样保留
    for f in _session_files(tmp_path, SID_DONE):
        assert f.exists()
    assert (tmp_path / "reports" / f"ResNet 复现_{SID_DONE}.md").exists()


def test_batch_delete_reports_result_and_resets_confirm(at, tmp_path):
    """删除后要给出可见的反馈，且确认框自动复位（避免下次一键误删）。"""
    at.checkbox(key=f"hist_chk_{SID_DONE}").set_value(True)
    at.run()
    at.checkbox(key="confirm_batch_del").set_value(True)
    at.run()
    at.button(key="batch_del_btn").click()
    at.run()
    assert not at.exception

    # 提示跨 rerun 存活（st.success 后紧跟 st.rerun 会被丢弃，故走 flash）
    assert any("已批量删除 1 个会话" in s.value for s in at.success)
    assert at.checkbox(key="confirm_batch_del").value is False


# ---------------- 依赖缓存面板 ----------------

DEPS_HOT = "a" * 16
DEPS_COLD = "b" * 16


def _mk_deps_dir(root: Path, name: str, last_used: str, packages=("numpy",)):
    d = root / name
    (d / "somepkg").mkdir(parents=True, exist_ok=True)
    (d / "somepkg" / "__init__.py").write_text("x" * 64, encoding="utf-8")
    for pkg in packages:
        (d / f"{pkg}-1.0.dist-info").mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps({"kind": "reqs", "installed_at": last_used,
                    "last_used": last_used, "requirements": "numpy"}),
        encoding="utf-8")


@pytest.fixture
def at_with_deps(tmp_path, monkeypatch):
    """带两个依赖缓存目录（一热一冷）的 AppTest。"""
    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    deps_root = tmp_path / "deps"
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(deps_root))
    _mk_deps_dir(deps_root, DEPS_HOT, "2026-09-19T10:00:00")
    _mk_deps_dir(deps_root, DEPS_COLD, "2026-01-01T10:00:00")

    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()
    return app


def test_deps_cache_panel_lists_entries(at_with_deps, tmp_path):
    """依赖缓存要在「清理管理」里可见（含包名/占用/总占用）。"""
    assert not at_with_deps.exception
    assert set(at_with_deps.multiselect(key="deps_del_pick").options) == {
        DEPS_HOT, DEPS_COLD}
    assert "2 个目录" in _captions(at_with_deps)


def test_deps_cache_delete_requires_pick_and_confirmation(at_with_deps, tmp_path):
    """未选目录/未勾确认时不可删；确认后才删得掉，且确认框自动复位。"""
    app = at_with_deps
    assert app.button(key="deps_del_btn").disabled is True

    app.multiselect(key="deps_del_pick").set_value([DEPS_COLD])
    app.run()
    assert app.button(key="deps_del_btn").disabled is True     # 仍未确认

    app.checkbox(key="confirm_deps_del").set_value(True)
    app.run()
    assert app.button(key="deps_del_btn").disabled is False
    assert (tmp_path / "deps" / DEPS_COLD).is_dir()           # 只渲染不点击

    app.button(key="deps_del_btn").click()
    app.run()
    assert not app.exception
    assert not (tmp_path / "deps" / DEPS_COLD).exists()
    assert (tmp_path / "deps" / DEPS_HOT).is_dir()            # 没选的不受牵连
    assert any("已删除 1 个依赖目录" in s.value for s in app.success)
    assert app.checkbox(key="confirm_deps_del").value is False


def test_select_all_visible_only_checks_visible(at):
    """「全选可见」只勾可见的那些，隐藏的既不被勾也不算进已选。"""
    at.selectbox(key="hist_state_filter").set_value("COMPLETED")
    at.run()
    # 被筛掉的行连勾选框都不渲染
    assert not [c for c in at.checkbox if c.key == f"hist_chk_{SID_RUNNING}"]

    at.button(key="hist_select_all").click()
    at.run()
    assert not at.exception
    assert "已选 1 条" in _captions(at)
    assert at.checkbox(key=f"hist_chk_{SID_DONE}").value is True

    # 筛回全部：RUNNING 那行的勾选框重新出现，且并未被「全选可见」勾上
    at.selectbox(key="hist_state_filter").set_value("全部")
    at.run()
    assert at.checkbox(key=f"hist_chk_{SID_RUNNING}").value is False

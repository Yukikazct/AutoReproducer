"""侧边栏 Docker 引擎门禁的端到端测试（AppTest 驱动真实 app.py）。

背景（用户实测踩到）：原判定是 `shutil.which("docker") is not None`——
只证明 **CLI 二进制在 PATH**，不证明 **Docker Desktop 的引擎在跑**。
于是侧边栏显示「✅ Docker 已就绪」，`docker run` 随后失败，报告里出现
`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`。

本文件锁死四件事：
1. 引擎不可用时**不谎报就绪**：文案给出原因 + 本地隔离兜底，开关拉回关闭；
2. 引擎可用时仍报「已就绪」，开关可用；
3. 「重新检测」按钮能刷新探测结果（启动 Docker Desktop 后无需刷新页面）；
4. Mock 模式**不做**探测（不执行代码、用不上 Docker，也不该让 Mock
   用例背上真实子进程探测）。

引擎探测本身经 `BaseAgent.docker_engine_available` 打桩——真实探测问的是
本机 Docker Desktop 在不在跑，用例结果不该随机器状态漂移；探测函数自身
的行为（version/info 选择、超时、无 CLI）由 tests/test_architecture.py 覆盖。

运行: python -m pytest tests/test_app_docker_gate.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base_agent import BaseAgent  # noqa: E402
import frontend.history_manager as hm  # noqa: E402

APP_PATH = str(Path(__file__).parent.parent / "app.py")
DOWN_REASON = "Docker 引擎未启动或不可用（daemon 连接失败）"


class _Probe:
    """可变的探测桩：记录调用次数，结果可中途翻转（模拟用户启动引擎）。"""

    def __init__(self, ok: bool = False):
        self.ok = ok
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return (True, None) if self.ok else (False, DOWN_REASON)


@pytest.fixture
def at_app(tmp_path, monkeypatch):
    """返回 (AppTest 工厂, 探测器)；数据目录指向 tmp_path。"""
    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    probe = _Probe()
    monkeypatch.setattr(BaseAgent, "docker_engine_available",
                        staticmethod(probe))

    def _make():
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(APP_PATH, default_timeout=120)

    return _make, probe


def _captions(at) -> str:
    return " ".join(c.value for c in at.sidebar.caption)


def _docker_toggle(at):
    return next(t for t in at.sidebar.toggle if "Docker" in t.label)


def _to_real_mode(at):
    """切到真实模式（Mock 开关是侧边栏第一个 toggle）。"""
    at.sidebar.toggle[0].set_value(False)
    at.run()
    assert not at.exception, at.exception


# ---------------- 1. 引擎不可用：不谎报就绪 ----------------

def test_engine_down_does_not_claim_ready(at_app):
    make, probe = at_app
    at = make()
    at.run()
    _to_real_mode(at)

    caps = _captions(at)
    assert "已就绪" not in caps, "引擎没在跑就不能报「已就绪」"
    assert "未启动" in caps                    # 原因是人话
    assert "本地隔离" in caps                  # 兜底去向明确


def test_engine_down_disables_toggle_and_forces_off(at_app):
    """显示开着却跑不了是最坏的组合：引擎不在时开关必须关且不可点。"""
    make, _ = at_app
    at = make()
    at.run()
    _to_real_mode(at)

    toggle = _docker_toggle(at)
    assert toggle.disabled is True
    assert toggle.value is False
    assert at.session_state["use_docker"] is False


# ---------------- 2. 引擎可用：仍然报就绪 ----------------

def test_engine_up_reports_ready_and_enables_toggle(at_app):
    make, probe = at_app
    probe.ok = True
    at = make()
    at.run()
    _to_real_mode(at)

    assert "已就绪" in _captions(at)
    toggle = _docker_toggle(at)
    assert toggle.disabled is False
    assert toggle.value is True


# ---------------- 3. 重新检测：启动引擎后无需刷新页面 ----------------

def test_recheck_button_refreshes_probe(at_app):
    make, probe = at_app
    at = make()
    at.run()
    _to_real_mode(at)
    assert "未启动" in _captions(at)
    first_calls = probe.calls

    # 用户去启动 Docker Desktop，然后点「重新检测」
    probe.ok = True
    recheck = next(b for b in at.sidebar.button if "重新检测" in b.label)
    recheck.click()
    at.run()

    assert not at.exception, at.exception
    assert probe.calls > first_calls, "重新检测必须真的重新探测"
    assert "已就绪" in _captions(at)
    assert _docker_toggle(at).disabled is False


# ---------------- 4. Mock 模式不探测 ----------------

def test_mock_mode_skips_probe(at_app):
    make, probe = at_app
    at = make()
    at.run()

    assert not at.exception, at.exception
    assert probe.calls == 0, "Mock 模式不该触发 Docker 探测"
    assert "Mock 模式不执行真实代码" in _captions(at)

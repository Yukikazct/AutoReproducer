"""Streamlit AppTest 冒烟测试：前端 LLM API 配置区与连接测试入口的渲染逻辑。

验证覆盖（对应需求"前端输入 API -> 调用 AI 复现"的界面层）：
- Mock 模式：页面加载无异常，配置输入禁用，连接测试按钮禁用并提示；
- 真实模式：三个配置输入（地址/Key/模型）可用，显示生效配置摘要（输入
  留空回退环境变量），连接测试按钮可用；
- 连接测试的真实调用链路由 tests/test_llm_config.py 单独覆盖（本地假
  OpenAI 服务，200/401/不可达三路径），此处不触发真实外网请求。
"""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_APP_PATH = str(__import__("pathlib").Path(__file__).parent.parent / "app.py")


def _app() -> AppTest:
    at = AppTest.from_file(_APP_PATH, default_timeout=120)
    # 预置 Docker 探测结果（(可用, 原因)）：本文件测的是 LLM 配置区渲染，
    # 不该随本机 Docker Desktop 是否在跑而变（真实模式下 app.py 会探测，
    # 引擎没起时侧边栏文案与按钮都会变）。探测本身由
    # tests/test_app_docker_gate.py 与 tests/test_architecture.py 覆盖。
    at.session_state["docker_probe"] = (True, None)
    return at


def test_mock_mode_panel_renders():
    at = _app()
    at.run()
    assert not at.exception, at.exception
    # Mock 开关默认开启
    assert len(at.sidebar.toggle) >= 1
    assert at.sidebar.toggle[0].value is True
    # 提示文案
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "Mock 模式不调用真实 LLM" in captions
    # 连接测试按钮存在且禁用
    btns = at.sidebar.button
    link = next(b for b in btns if "测试 AI 连接" in b.label)
    assert link.disabled is True


def test_real_mode_panel_enables_inputs_and_button():
    at = _app()
    at.run()
    at.sidebar.toggle[0].set_value(False)
    at.run()
    assert not at.exception, at.exception
    # 配置输入可用
    inputs = {i.label: i for i in at.sidebar.text_input}
    assert inputs["API 地址（OpenAI 兼容）"].disabled is False
    assert inputs["API Key"].disabled is False
    assert inputs["模型名称"].disabled is False
    # 生效配置摘要（默认回退值）
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "当前生效" in captions
    assert "deepseek-chat" in captions
    # 连接测试按钮可用
    btns = at.sidebar.button
    link = next(b for b in btns if "测试 AI 连接" in b.label)
    assert link.disabled is False
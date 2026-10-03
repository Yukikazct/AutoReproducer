"""前端 LLM 配置解析与连接测试（无 Streamlit 依赖，便于单元测试）。

职责：
1. resolve_llm_config()  —— 界面输入优先，缺省回退环境变量，返回生效配置；
2. config_missing()     —— 真实模式下关键配置（界面+环境变量均缺失）检查；
3. test_llm_connection()—— 真实调用一次 Chat Completions，验证 API 可用性，
                           返回 (ok, message)，供前端"测试连接"按钮与单测复用。

与 src.llm.llm_client 保持一致：OpenAI 兼容端点，环境变量
LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_TIMEOUT 为回退来源。
"""
import os

from src.llm.llm_client import LLMClient

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

CONNECT_TEST_TIMEOUT = 20  # 连接测试用较短超时，避免长时间挂起


def resolve_llm_config(base_url: str = "", api_key: str = "",
                       model: str = "") -> dict:
    """解析最终生效的 LLM 配置：界面输入 > 环境变量 > 内置默认值。"""
    return {
        "base_url": (base_url.strip()
                     or os.environ.get("LLM_BASE_URL", "").strip()
                     or DEFAULT_BASE_URL),
        "api_key": api_key.strip()
                   or os.environ.get("LLM_API_KEY", "").strip(),
        "model": (model.strip()
                  or os.environ.get("LLM_MODEL", "").strip()
                  or DEFAULT_MODEL),
    }


def config_missing(base_url: str = "", model: str = "") -> list:
    """返回真实模式下缺失的关键配置项（界面输入与环境变量均无）。

    返回空列表表示配置就绪；否则为缺失项的人类可读名称列表。
    """
    missing = []
    if not (base_url.strip() or os.environ.get("LLM_BASE_URL", "").strip()):
        missing.append("API 地址 (LLM_BASE_URL)")
    if not (model.strip() or os.environ.get("LLM_MODEL", "").strip()):
        missing.append("模型名称 (LLM_MODEL)")
    return missing


def test_llm_connection(base_url: str = "", api_key: str = "",
                        model: str = "", timeout: int = CONNECT_TEST_TIMEOUT,
                        prompt: str = "连接测试：请回复 OK。"
                        ) -> tuple:
    """用给定配置真实调用一次 LLM，验证 API 可用性。

    Returns:
        (True, 成功消息)  —— 真实拿到非错误响应；
        (False, 错误消息) —— 未配置 / HTTP 错误 / 网络错误 / 异常响应。
    """
    cfg = resolve_llm_config(base_url, api_key, model)
    try:
        llm = LLMClient(mock_mode=False,
                        base_url=cfg["base_url"],
                        api_key=cfg["api_key"],
                        model=cfg["model"],
                        timeout=timeout,
                        max_tokens=64)
        resp = (llm.chat(prompt, temperature=0.1) or "").strip()
    except Exception as exc:
        return False, f"[LLM API Error: {exc}]"
    if resp.startswith("[LLM API Error"):
        return False, resp
    preview = resp[:100]
    if not preview:
        return False, ("[LLM API Error: 响应内容为空——请检查模型是否可用"
                       "以及返回格式是否兼容 OpenAI Chat Completions]")
    return True, f"连接成功 ({cfg['model']} @ {cfg['base_url']}): {preview}"

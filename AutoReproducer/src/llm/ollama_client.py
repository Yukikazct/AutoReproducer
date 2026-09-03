"""Ollama LLM客户端 - 封装的Ollama API调用"""
import json
import urllib.request
import urllib.error
from typing import Optional


class LLMClient:
    """Ollama API客户端，支持Mock模式和真实模式"""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b",
                 mock_mode: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mock_mode = mock_mode
        self.call_count = 0

    def chat(self, prompt: str, system_prompt: str = "",
             temperature: float = 0.3) -> str:
        """发送聊天请求到Ollama"""
        self.call_count += 1

        if self.mock_mode:
            return self._mock_response(prompt)

        return self._real_chat(prompt, system_prompt, temperature)

    def _real_chat(self, prompt: str, system_prompt: str,
                   temperature: float) -> str:
        """真实的Ollama API调用"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "temperature": temperature
        }
        if system_prompt:
            payload["messages"].append({
                "role": "system",
                "content": system_prompt
            })
        payload["messages"].append({
            "role": "user",
            "content": prompt
        })

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("message", {}).get("content", "")
        except Exception as e:
            return f"[Ollama API Error: {e}]"

    def _mock_response(self, prompt: str) -> str:
        """Mock模式返回模拟数据"""
        if "paper" in prompt.lower() or "论文" in prompt:
            return json.dumps({
                "title": "AutoReproducer: 基于多智能体协作的论文自动复现与优化系统",
                "authors": ["AutoReproducer Team"],
                "method": "多智能体协作框架，包含1个Orchestrator和7个Agent",
                "dependencies": ["Python 3.12", "PyTorch", "Streamlit"],
                "metrics": {"accuracy": 0.85, "f1_score": 0.82},
                "dataset": "CIFAR-10",
                "code_url": "https://github.com/example/autoreproducer"
            })
        elif "environment" in prompt.lower() or "环境" in prompt:
            return json.dumps({
                "required_packages": ["torch>=2.0.0", "torchvision>=0.15.0",
                                      "numpy>=1.24.0", "tqdm>=4.65.0"],
                "python_version": "3.12",
                "cuda_version": "12.1"
            })
        elif "code" in prompt.lower() or "代码" in prompt:
            return "print('Training complete. Test accuracy: 85.2%')"
        elif "optimize" in prompt.lower() or "优化" in prompt:
            return json.dumps({
                "suggestions": [
                    "将学习率从0.01降至0.005",
                    "增加Batch Normalization层",
                    "使用AdamW优化器替代SGD"
                ]
            })
        else:
            return f"[Mock Response] 收到请求: {prompt[:50]}..."

    def get_call_count(self) -> int:
        return self.call_count
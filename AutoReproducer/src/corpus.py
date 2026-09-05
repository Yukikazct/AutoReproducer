"""语料对照层 - 读取 PaperGuru-Benchmark 的 23 个真实复现仓库作为轻量锚点。

所有函数在语料目录缺失或解析失败时优雅降级(返回空值),不影响 Mock 演示。
"""
import json
from pathlib import Path

# src/corpus.py -> parents[0]=src, parents[1]=AutoReproducer, parents[2]=仓库根目录
_PAPERBENCH_DIR = (
    Path(__file__).resolve().parents[2] / "PaperGuru-Benchmark" / "PaperBench"
)


def _aggregate() -> dict:
    p = _PAPERBENCH_DIR / "aggregate-final.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def list_papers() -> list:
    """返回 23 篇论文 id 及复现分,如 [{"id": "bbox", "score": 0.403}, ...]。"""
    per_paper = _aggregate().get("per_paper", {})
    return [{"id": pid, "score": score} for pid, score in per_paper.items()]


def get_declared_score(paper_id: str):
    """返回某论文的复现分(0-1);不存在返回 None。"""
    return _aggregate().get("per_paper", {}).get(paper_id)


def get_requirements(paper_id: str) -> str:
    """读取某论文的 requirements.txt 内容;不存在返回空字符串。"""
    p = _PAPERBENCH_DIR / "submissions" / paper_id / "submission" / "requirements.txt"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def get_reproduce_script(paper_id: str) -> str:
    """读取某论文的一键复现脚本;不存在返回空字符串。"""
    p = _PAPERBENCH_DIR / "submissions" / paper_id / "submission" / "reproduce.sh"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""

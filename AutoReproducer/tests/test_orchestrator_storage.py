"""Orchestrator 三层存储接入测试（P0-2）。

覆盖：
1. paper_id 生成：corpus_paper 优先，否则 sha1(title) 前 12 位；
2. FIND_RESOURCES 后 fetch 钩子：code/dataset/weights 懒加载结果
   进入 data.storage.fetched（state 合理，失败不抛异常）；
3. COMPLETED 后 manifest 落盘：文件存在、字段齐全、stats 聚合；
4. fetch 异常不阻断流水线（mock 抛错仍返回 COMPLETED）；
5. 占位引用归一化（'未找到' -> 空，不触发下载）。

运行: python -m pytest tests/test_orchestrator_storage.py -v
"""
from pathlib import Path

import pytest

if str(Path(__file__).parent.parent) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(Path(__file__).parent.parent))

from src.audit.audit_logger import AuditLogger  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402
from src.resource_manager import ResourceManager  # noqa: E402


@pytest.fixture()
def orch(tmp_path):
    """隔离数据根的完整 Orchestrator（mock 模式，不触网）。"""
    rm = ResourceManager(data_root=str(tmp_path / "data"))
    return Orchestrator(llm_client=LLMClient(mock_mode=True),
                        mock_mode=True, logger=AuditLogger(),
                        resource_manager=rm)


# ---------------- 1. paper_id 生成 ----------------

def test_paper_id_from_corpus_key(orch):
    result = orch.run({"paper_title": "ResNet: Deep Residual Learning",
                       "corpus_paper": "corpus_resnet50"})
    storage = result["data"]["storage"]
    assert storage["paper_id"] == "corpus_resnet50"


def test_paper_id_from_title(orch):
    result = orch.run({"paper_title": "ResNet: Deep Residual Learning"})
    pid = result["data"]["paper_id"]
    assert len(pid) == 12
    # 同标题稳定生成同一 ID
    assert pid == ResourceManager.paper_id_for(
        "ResNet: Deep Residual Learning")


# ---------------- 2. fetch 钩子 ----------------

def test_fetch_hook_populates_storage(orch):
    result = orch.run({"paper_title": "ResNet: Deep Residual Learning"})
    assert result["state"] == "COMPLETED", result.get("error")
    fetched = result["data"]["storage"]["fetched"]
    assert set(fetched) == {"code", "dataset", "weights"}
    # mock finder 输出占位 URL -> code 不下载；dataset 合成冒烟集
    assert fetched["code"]["state"] in ("placeholder-skip", "skipped")
    assert fetched["dataset"]["state"] in ("smoke-synth", "cached")
    assert fetched["weights"]["state"] in ("none", "skipped")


# ---------------- 3. manifest 落盘 ----------------

def test_finalize_writes_manifest(orch, tmp_path):
    result = orch.run({"paper_title": "ResNet: Deep Residual Learning"})
    storage = result["data"]["storage"]
    pid = storage["paper_id"]
    manifest_path = Path(storage["manifest_path"])
    assert manifest_path.is_file()
    manifest = storage["manifest"]
    assert manifest["paper_id"] == pid
    assert manifest["paper_title"] == "ResNet: Deep Residual Learning"
    assert set(manifest["resources"]) == {"code", "dataset", "weights"}
    assert storage["stats"]["usage"]["bytes"] >= 0
    assert storage["stats"]["manifest_count"] >= 1
    # manifest 可被 ResourceManager 独立读取（存量格式兼容）
    assert orch.resource_manager.get_manifest(pid) is not None
    # 合成冒烟集确实落盘
    assert (tmp_path / "data" / "datasets" / pid /
            "dataset_smoke" / "samples.csv").is_file()


def test_manifest_idempotent_rerun(orch, tmp_path):
    """同一标题跑两次：manifest 覆盖更新，不产生重复条目。"""
    orch.run({"paper_title": "ResNet: Deep Residual Learning"})
    orch2 = Orchestrator(llm_client=LLMClient(mock_mode=True),
                         mock_mode=True, logger=AuditLogger(),
                         resource_manager=ResourceManager(
                             data_root=str(tmp_path / "data")))
    orch2.run({"paper_title": "ResNet: Deep Residual Learning"})
    stats = orch2.resource_manager.stats()
    assert stats["manifest_count"] == 1  # 同一 paper_id 只落一份


# ---------------- 4. fetch 异常不阻断 ----------------

def test_fetch_failure_does_not_block_pipeline(orch, monkeypatch):
    def boom(self, *a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(orch.resource_manager, "fetch_code", boom)
    result = orch.run({"paper_title": "ResNet: Deep Residual Learning"})
    assert result["state"] == "COMPLETED"
    assert "fetched" not in result["data"]["storage"]


def test_finalize_failure_does_not_block_pipeline(orch, monkeypatch):
    def boom(self, *a, **kw):
        raise RuntimeError("disk full")

    orch2 = orch
    monkeypatch.setattr(orch2.resource_manager, "save_manifest", boom)
    result = orch2.run({"paper_title": "ResNet: Deep Residual Learning"})
    assert result["state"] == "COMPLETED"


# ---------------- 5. 占位引用归一化 ----------------

def test_clean_ref_normalization():
    assert Orchestrator._clean_ref("未找到") == ""
    assert Orchestrator._clean_ref("未知") == ""
    assert Orchestrator._clean_ref("none") == ""
    assert Orchestrator._clean_ref("  ") == ""
    assert Orchestrator._clean_ref("https://github.com/real/repo") == \
        "https://github.com/real/repo"
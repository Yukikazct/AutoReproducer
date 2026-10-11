"""ReZero reports distinguish source review, completed execution and accuracy."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src import method_advice
from src.agents.report_generator import ReportGeneratorAgent
from src.repository_profiles import get_profile
from src.repository_routing import _REVIEWED_REPOSITORIES, REZERO_DISCOVERY_REPOSITORY


def profile():
    return get_profile("rezero_cifar10_reference")


def test_rezero_online_review_uses_actual_components_and_metrics_without_claiming_results(monkeypatch):
    frozen = profile()
    original = deepcopy(frozen)
    author = {"source_id": "author_training", "origin": "official_repository",
              "url": frozen["paper"]["reference_source"], "locator": "train_faster_superc.py",
              "content": "seed = 6892\nn_epoch = 45\nmodel = rezero_preactresnet18()"}
    project = {"source_id": "independent_evaluator", "origin": "project_adapter",
               "url": "project://evaluate.py", "locator": "evaluate.py",
               "content": "metrics = {'top1_accuracy_pct': accuracy, 'cross_entropy': loss}"}
    prompts, contexts = [], []
    def request(llm, prompt, timeout):
        prompts.append(prompt)
        context = json.loads(prompt.split("\n", 1)[1])
        contexts.append(context)
        evidence = [{"source_id": "author_training", "quote": "seed = 6892"},
                    {"source_id": "author_training", "quote": "n_epoch = 45"}]
        if context["review_responsibility"] == "project_adaptation_readiness":
            evidence.append({"source_id": "independent_evaluator", "quote": project["content"]})
        return {"response": json.dumps({"status": "accepted", "summary": "依据来源核对，训练尚未开始。",
                                        "evidence": evidence})}
    monkeypatch.setattr(method_advice, "request_text", request)
    llm = SimpleNamespace(mock_mode=False, base_url="https://example.org", model="test")
    analysis = method_advice.review_sources(llm, frozen, [author], lambda *args: None, project_sources=[project])
    assert analysis["status"] == "accepted" and len(contexts) == 4
    assert frozen == original
    for context, prompt in zip(contexts, prompts):
        scope = context["experiment_scope"]
        assert scope["kind"] == "selected_paper_experiment"
        assert scope["reference_kind"] == "author_notebook"
        assert scope["paper_full_text_provided"] is False
        assert scope["paper_table_reproduction"] is False
        assert scope["execution_completed"] is False
        assert scope["excluded_experiments"] == ["enwiki8", "whole_paper"]
        assert "MAE/RMSE" not in prompt and "PSNR" not in prompt
        assert "author_model.py" not in prompt and "explicit_solver_options" not in prompt
    for context in contexts[:2]:
        assert context["sources"] == [author]
        assert "experiment_contract" not in context and "project_adaptations" not in context
    for context in contexts[2:]:
        adaptations = context["project_adaptations"]
        assert "models/rezero_preact_resnet.py" in adaptations["author_component_loading"]
        assert "customonecycle.py" in adaptations["author_component_loading"]
        assert "10,000" in adaptations["independent_evaluation"]
        assert "top1_accuracy_pct" in adaptations["independent_evaluation"]
        assert "cross_entropy" in adaptations["independent_evaluation"]
        assert "训练尚未开始" in adaptations["execution_state"]
    assert analysis["input_scope"]["user_pdf_shared"] is False


def run_data(accuracy=94.0):
    frozen = profile()
    passed = accuracy >= 94.0
    route = deepcopy(_REVIEWED_REPOSITORIES[REZERO_DISCOVERY_REPOSITORY.casefold()])
    return {"experiment_spec": frozen, "paper_info": frozen["paper"],
            "execution": {"mode": "repository", "executed": True, "success": True,
                          "final": {"exit_code": 0, "success": True, "stdout": "independent evaluation complete"}},
            "validation": {"status": "reproduced" if passed else "reference_not_met",
                           "is_reproduced": passed, "result_level": "reproduced" if passed else "experiment_completed",
                           "scope": "selected_paper_experiment", "protocol_pass": True,
                           "independent_metrics_pass": True, "quality_pass": passed,
                           "metrics_comparison": {"paper": {"top1_accuracy_pct": 94.0},
                                                  "actual": {"top1_accuracy_pct": accuracy, "cross_entropy": .321}},
                           "training_summary": {"epochs_completed": 45, "steps_completed": 4410,
                                                "train_samples": 50000, "test_samples": 10000,
                                                "best_epoch": 44, "best_accuracy_pct": accuracy}},
            "resources": {"code_repo_url": route["training_repository_url"],
                          "training_repository_url": route["training_repository_url"],
                          "discovery_repository_url": route["discovery_repository_url"],
                          "repository_relationship": route["repository_relationship"],
                          "selection_evidence": {"source": "pdf_text", "page": 4,
                                                 "url": REZERO_DISCOVERY_REPOSITORY,
                                                 "context": "Code for ReZero is available at " + REZERO_DISCOVERY_REPOSITORY,
                                                 "evidence_type": "author_code_statement"}},
            "pdf_resolution": {"sha256": "54eea1741fad2e7a20e52f55c64b34977f1206c5e3c8e14804ea5385c299c070"},
            "optimization": {"mode": "off", "available": False, "optimized": False},
            "method_analysis": {"status": "accepted", "reviews": [
                {"role": "reader", "summary": "已核对作者训练入口。", "evidence": [
                    {"source_id": "author_training", "quote": "n_epoch = 45"}]}]}}


@pytest.mark.parametrize("accuracy,expected", [(94.0, "选定论文实验数值复现通过"),
                                               (93.91, "选定论文实验已完成，数值参考未达到")])
def test_rezero_report_preserves_execution_success_and_strict_independent_comparison(accuracy, expected):
    data = run_data(accuracy)
    report = ReportGeneratorAgent()._build_report(data)
    assert expected in report
    assert "❌ 执行失败" not in report and "❌ 失败" not in report
    assert "45 轮 / 4410 个优化器步骤" in report
    assert "训练 50000 / 测试 10000" in report
    assert "第 44 轮" in report
    assert f"top-1 {accuracy:.2f}% / 固定参考 94.00%" in report
    assert "cross_entropy 0.321000" in report
    assert f"| top1_accuracy_pct | 94.0% | {accuracy}% |" in report
    assert "| 指标 | 作者公开参考 | 独立评估实测 |" in report
    assert "不覆盖 enwiki8 或整篇论文" in report
    assert "PyTorch 2.5.1" in report and "PyTorch 1.2" in report
    assert "在线来源分析：reader" in report and "已核对作者训练入口" in report
    assert "本实验按冻结作者协议运行" in report
    assert "没有对应论文表格数值验收" not in report
    if accuracy < 94:
        assert "数值复现通过" not in report


def test_rezero_report_renders_pdf_and_complete_immutable_author_link_chain():
    data = run_data()
    report = ReportGeneratorAgent()._build_report(data)
    assert "**论文原文发现仓库**: https://github.com/majumderb/rezero" in report
    assert "**实际固定训练仓库**: https://github.com/tbachlechner/ReZero-Superconvergence" in report
    assert "第 4 页" in report and data["pdf_resolution"]["sha256"] in report
    for index, hop in enumerate(data["resources"]["repository_relationship"]["hops"], 1):
        assert f"作者链接第 {index} 跳" in report
        assert hop["source_url"] in report
        assert hop["target_repository_url"] in report
        assert hop["sha256"] in report
    assert "ReZero-examples" in report


@pytest.mark.parametrize("status", ["prepared", "environment_prepared", "analysis_failed"])
def test_untrained_rezero_report_keeps_expected_protocol_distinct_from_actual_evidence(status):
    data = run_data()
    data["execution"].update(executed=False, success=False, final={})
    data["validation"] = {"status": status, "is_reproduced": None}
    if status == "analysis_failed":
        data["method_analysis"].update(status="rejected", failed_role="finder", reason="缺少来源映射")
    report = ReportGeneratorAgent()._build_report(data)
    assert "**冻结训练协议**" in report
    assert "尚无完成协议核验的训练摘要" in report
    assert "**实际完成**" not in report and "**独立重载评估**" not in report
    assert "数值复现通过" not in report
    if status == "analysis_failed":
        assert "**失败阶段**: 资源核对" in report
        assert "训练尚未开始" in report and "method_analysis.json" in report

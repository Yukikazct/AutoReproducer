"""Repository analysis reports keep model readiness and measured verdicts separate."""
from copy import deepcopy

import pytest

from src.agents.report_generator import ReportGeneratorAgent


@pytest.fixture
def report_data():
    source = {"source_id": "paper_table2", "locator": "Table 2",
              "url": "https://arxiv.org/html/2205.13504v3#S5.T2"}
    ref = {"source_id": source["source_id"], "locator": source["locator"],
           "quote": "DLinear ETTh1 96 MSE 0.375 MAE 0.399"}
    stages = [{"name": name, "attempted": True, "completed": True, "accepted": True,
               "calls": 1, "usage": {"total_tokens": 70}} for name in ("reader", "finder", "builder", "verifier")]
    readiness = {"status": "accepted", "calls": 4, "model": "actual-configured-model", "stages": stages,
                 "sources": [source], "gate": {"pass": True}, "analyses": {
        "reader": {"protocol": {"seq_len": 336, "pred_len": 96, "features": "M"},
                   "reference_metrics": {"mse": 0.375, "mae": 0.399},
                   "evidence": {"protocol.seq_len": [ref]}},
        "finder": {"entrypoints": {"model": "models/DLinear.py", "training_and_test": "exp/exp_main.py"},
                   "evidence": {"entrypoints.model": [ref]}},
        "builder": {"original_requirements": ["numpy", "torch==1.9.0"],
                    "compatibility_note": "Modern Python requires a separately tested compatibility environment.",
                    "compatibility_proposals": [{"package": "torch", "suggested_constraint": ">=2.0", "reason": "Verify modern Python wheel support."}],
                    "proposals_executed": False, "evidence": {"original_requirements": [ref]}},
        "verifier": {"pass": True, "checks": {"protocol_alignment": True}, "issues": [],
                     "evidence": {"checks.protocol_alignment": [ref]}},
    }}
    result_analysis = {"status": "accepted", "calls": 1, "summary": "The selected experiment has close numerical results.",
                       "stages": [{"name": "result_validator", "attempted": True, "completed": True,
                                   "accepted": True, "calls": 1, "usage": {"total_tokens": 42}}],
                       "differences": [{"metric": "mse", "paper_value": 0.375, "actual_value": 0.3841444,
                                        "absolute_difference": 0.0091444, "relative_difference": 0.0243851,
                                        "explanation": "The measured MSE is higher."}],
                       "limitations": ["Only one selected experiment is covered."],
                       "evidence": {"summary": [ref], "differences.mse": [ref], "limitations.0": [ref]}}
    return {"paper_info": {"title": "Are Transformers Effective for Time Series Forecasting?", "method": "DLinear", "dataset": "ETTh1"},
            "execution": {"mode": "repository", "executed": True, "final": {"success": True, "exit_code": 0, "stdout": "REAL_TRAINING_LOG"}},
            "validation": {"status": "reproduced", "is_reproduced": True,
                           "validation": {"verdict_source": "deterministic", "relative_tolerance": 0.05}},
            "repository_analysis": readiness, "result_analysis": result_analysis, "analysis_status": "completed"}


def build(data):
    return ReportGeneratorAgent()._build_report(data)


def test_report_shows_distinct_specialists_actual_calls_and_accepted_evidence(report_data):
    report = build(report_data)
    assert "公开来源多 Agent 分析" in report
    assert "真实调用数**: 4" in report
    for agent in ("PaperReader", "ResourceFinder", "EnvBuilder", "Verifier"):
        assert f"| {agent} | 是 | 是 | 是 | 1 | 70 |" in report
    assert "| seq\\_len | 336 |" in report
    assert "| pred\\_len | 96 |" in report
    assert "models/DLinear.py" in report
    assert "DLinear ETTh1 96 MSE 0.375 MAE 0.399" in report
    assert "[paper\\_table2 · Table 2](<https://arxiv.org/html/2205.13504v3#S5.T2>)" in report
    assert "最终数值结论由本地确定性核验给出" in report
    assert "不代表论文全部实验" in report


def test_report_distinguishes_original_requirements_and_unexecuted_proposals(report_data):
    report = build(report_data)
    assert "作者原始依赖**: numpy, torch==1.9.0" in report
    assert "Modern Python requires a separately tested compatibility environment." in report
    assert "兼容候选状态**: 未执行" in report
    assert "torch\\>=2.0: Verify modern Python wheel support." in report
    assert "准备条件审查**: 通过" in report


def test_result_report_explains_accepted_summary_with_one_real_call_and_scope(report_data):
    report = build(report_data)
    assert "LLM 分析状态**: 已完成" in report
    assert "| ResultValidator | 是 | 是 | 是 | 1 | 42 |" in report
    assert "The selected experiment has close numerical results." in report
    assert "| mse | 0.375 | 0.3841444 | 0.0091444 | 2.44% |" in report
    assert "Only one selected experiment is covered." in report
    assert "用户授权的 MSE、MAE、已完成轮数、协议核验与独立复算状态摘要" in report
    assert "选定论文实验数值复现通过" in report
    assert "最终结论由确定性核验给出" in report


def test_failed_result_api_cannot_replace_successful_training_and_verdict(report_data):
    report_data["result_analysis"] = {"status": "failed", "calls": 1,
        "stages": [{"name": "result_validator", "attempted": True, "completed": False, "accepted": False, "calls": 1, "usage": {}}],
        "gate": {"pass": False, "reason": "API request timed out"}}
    report_data["analysis_status"] = "result_analysis_failed"
    report_data["analysis_error"] = "API request timed out"
    report = build(report_data)
    assert "LLM 分析状态**: 结果摘要解释失败" in report
    assert "| ResultValidator | 是 | 否 | 否 | 1 | 未返回 |" in report
    assert "API request timed out" in report
    assert "执行状态**: ✅ 成功" in report
    assert "选定论文实验数值复现通过" in report
    assert "API 解释失败不会覆盖已完成实验的确定性结论" in report


def test_rejected_readiness_is_unstarted_training_not_failed_training(report_data):
    report_data["repository_analysis"] = {"status": "failed", "model": "actual-configured-model", "calls": 1,
        "sources": [], "analyses": {}, "gate": {"pass": False, "reason": "No public evidence for input window"},
        "stages": [{"name": "reader", "attempted": True, "completed": True, "accepted": False, "calls": 1, "usage": {"total_tokens": 11}}]}
    report_data["execution"] = {"mode": "repository", "executed": False, "not_runnable": True,
                                "reason": "Public evidence gate did not pass", "final": {}}
    report_data["validation"] = {"status": "analysis_failed", "is_reproduced": False}
    report_data["analysis_status"] = "failed"
    del report_data["result_analysis"]
    report = build(report_data)
    assert "| PaperReader | 是 | 是 | 否 | 1 | 11 |" in report
    assert "No public evidence for input window" in report
    assert "未进入训练（公开协议分析未通过）" in report
    assert "执行状态**: ⚠️ 未运行" in report
    assert "❌ 执行失败" not in report
    assert "选定论文实验数值复现通过" not in report


def test_quote_link_requires_source_id_and_exact_locator(report_data):
    analysis = report_data["repository_analysis"]
    unknown = deepcopy(analysis["analyses"]["reader"]["evidence"]["protocol.seq_len"][0])
    unknown["locator"] = "Other table"
    analysis["analyses"] = {"reader": {"protocol": {}, "evidence": {"protocol.seq_len": [unknown]}}}
    del report_data["result_analysis"]
    report = build(report_data)
    assert "引用定位未匹配来源清单" in report
    assert "paper\\_table2 · Other table" in report
    assert "](<https://arxiv.org/html/2205.13504v3#S5.T2>)" not in report


def test_report_uses_actual_retry_count_and_can_show_without_result_opt_in(report_data):
    report_data["repository_analysis"]["calls"] = 5
    report_data["repository_analysis"]["stages"][0]["calls"] = 2
    report_data["analysis_status"] = "public_readiness_accepted"
    del report_data["result_analysis"]
    report = build(report_data)
    assert "真实调用数**: 5" in report
    assert "| PaperReader | 是 | 是 | 是 | 2 | 70 |" in report
    assert "未执行结果摘要解释" in report
    assert "| ResultValidator" not in report


def test_empty_analysis_does_not_invent_specialist_calls():
    report = build({"execution": {"mode": "repository", "final": {"success": True}}})
    assert "公开来源多 Agent 分析" not in report
    assert "真实调用数" not in report
    assert "LLM 分析状态与结果摘要解释" not in report


def test_model_text_is_literal_and_report_input_is_not_mutated(report_data):
    report_data["result_analysis"]["summary"] = "![do not embed](private.png) | fabricated **format**"
    before = deepcopy(report_data)
    report = build(report_data)
    assert "![do not embed](private.png)" not in report
    assert "\\!\\[do not embed\\]\\(private.png\\)" in report
    assert report_data == before


@pytest.mark.parametrize("stage,agent", [("reader", "PaperReader"), ("finder", "ResourceFinder"),
                                        ("builder", "EnvBuilder"), ("verifier", "Verifier")])
def test_each_rejected_specialist_report_preserves_typed_diagnostics_and_valid_quotes(
        report_data, stage, agent):
    analysis = report_data["repository_analysis"]
    ref = deepcopy(analysis["analyses"]["reader"]["evidence"]["protocol.seq_len"][0])
    analysis.update(status="failed", calls=1, analyses={},
                    gate={"pass": False, "reason": "Public readiness was rejected"},
                    stages=[{"name": stage, "attempt": 1, "attempted": True, "completed": True,
                             "accepted": False, "calls": 1,
                             "reason": "Source evidence did not pass local validation",
                             "rejection_diagnostics": {
                                 "status": "rejected", "pass": False,
                                 "reviewed_stages": ["reader", "finder", "builder"],
                                 "checks": {"protocol_alignment": True, "dependency_provenance": False},
                                 "issues": ["The modern environment has not been verified."],
                                 "evidence": {"checks.dependency_provenance": [ref]},
                                 "raw_response": "UNRETAINED_RAW_MODEL_RESPONSE"}}])
    report_data["execution"] = {"mode": "repository", "executed": False, "not_runnable": True, "final": {}}
    report_data["validation"] = {"status": "analysis_failed", "is_reproduced": False}
    report_data["analysis_status"] = "failed"
    del report_data["result_analysis"]
    before = deepcopy(report_data)
    report = build(report_data)
    assert f"{agent}：分析未接受（第 1 次）" in report
    assert "Source evidence did not pass local validation" in report
    assert "模型返回状态**: rejected" in report
    assert "模型自查结论**: 未通过" in report
    assert "| protocol\\_alignment | 通过 |" in report
    assert "| dependency\\_provenance | 未通过 |" in report
    assert "The modern environment has not been verified." in report
    assert "DLinear ETTh1 96 MSE 0.375 MAE 0.399" in report
    assert "[paper\\_table2 · Table 2](<https://arxiv.org/html/2205.13504v3#S5.T2>)" in report
    assert "未进入训练（公开协议分析未通过）" in report
    assert "选定论文实验数值复现通过" not in report
    assert "UNRETAINED_RAW_MODEL_RESPONSE" not in report
    assert report_data == before


def test_repaired_stage_keeps_earlier_rejection_separate_from_final_readiness_and_verdict(report_data):
    stages = report_data["repository_analysis"]["stages"]
    rejected = {**deepcopy(stages[0]), "attempt": 1, "accepted": False,
                "rejection_diagnostics": {"status": "accepted",
                                          "issues": ["The first quotation did not support the input length."]}}
    stages[0]["attempt"] = 2
    stages.insert(0, rejected)
    report_data["repository_analysis"]["calls"] = 5
    report = build(report_data)
    assert "PaperReader：分析未接受（第 1 次）" in report
    assert "The first quotation did not support the input length." in report
    assert "模型返回状态**: accepted" in report
    assert "真实调用数**: 5" in report
    assert "选定论文实验数值复现通过" in report
    assert "分析状态**: 通过" in report


def test_rejection_issue_markdown_is_literal_and_wrong_locator_never_gets_public_link(report_data):
    analysis = report_data["repository_analysis"]
    ref = deepcopy(analysis["analyses"]["reader"]["evidence"]["protocol.seq_len"][0])
    ref["locator"] = "Other table"
    analysis["analyses"] = {}
    analysis["stages"] = [{"name": "reader", "accepted": False,
                           "rejection_diagnostics": {
                               "issues": ["![injected](outside.png) | **untrusted markup**"],
                               "evidence": {"protocol.seq_len": [ref]}}}]
    del report_data["result_analysis"]
    report = build(report_data)
    assert "![injected](outside.png)" not in report
    assert "\\!\\[injected\\]\\(outside.png\\) \\| \\*\\*untrusted markup\\*\\*" in report
    assert "paper\\_table2 · Other table（引用定位未匹配来源清单）" in report
    assert "](<https://arxiv.org/html/2205.13504v3#S5.T2>)" not in report


def provenance_fixture(report_data):
    analysis = report_data["repository_analysis"]
    reader = analysis["analyses"]["reader"]
    sha = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
    new_sources = [
        {"source_id": "paper_implementation_b2", "locator": "#A2.SS2",
         "url": "https://arxiv.org/html/2205.13504v3#A2.SS2"},
        {"source_id": "repo_author_command", "locator": "scripts/EXP-LongForecasting/Linear/etth1.sh#L1-L80",
         "url": f"https://github.com/cure-lab/LTSF-Linear/blob/{sha}/scripts/EXP-LongForecasting/Linear/etth1.sh#L1-L80"},
        {"source_id": "repo_entrypoint", "locator": "run_longExp.py#L1-L200",
         "url": f"https://github.com/cure-lab/LTSF-Linear/blob/{sha}/run_longExp.py#L1-L200"},
        {"source_id": "author_single_seed", "locator": "#issuecomment-1331937601",
         "url": "https://github.com/cure-lab/LTSF-Linear/issues/33#issuecomment-1331937601"},
    ]
    analysis["sources"].extend(new_sources)
    sources = {source["source_id"]: source for source in analysis["sources"]}

    def origin(source_id, category):
        return {"source_id": source_id, "locator": sources[source_id]["locator"], "origin": category}

    reader["protocol"].update(seed=2021, train_epochs=10)
    reader["protocol_provenance"] = {
        "protocol.seq_len": {"value_sources": [origin("paper_implementation_b2", "paper_text"),
                                                 origin("repo_author_command", "author_script")], "context_sources": []},
        "protocol.seed": {"value_sources": [origin("repo_entrypoint", "author_code_setting_or_default")],
                          "context_sources": [origin("author_single_seed", "author_statement")]},
        "protocol.train_epochs": {"value_sources": [origin("repo_entrypoint", "author_code_setting_or_default")],
                                  "context_sources": []},
        "reference_metrics.mse": {"value_sources": [origin("paper_table2", "paper_text")], "context_sources": []},
        "reference_metrics.mae": {"value_sources": [origin("paper_table2", "paper_text")], "context_sources": []},
    }
    return reader, sources


def test_protocol_report_distinguishes_paper_script_code_defaults_and_statement_context(report_data):
    provenance_fixture(report_data)
    before = deepcopy(report_data)
    report = build(report_data)
    parameter_row = next(line for line in report.splitlines() if line.startswith("| seq\\_len |"))
    assert "336" in parameter_row and "论文原文 · paper\\_implementation\\_b2" in parameter_row
    assert "作者实验脚本 · repo\\_author\\_command" in parameter_row
    seed_row = next(line for line in report.splitlines() if line.startswith("| seed |"))
    assert "2021" in seed_row and "作者代码设置或默认值 · repo\\_entrypoint" in seed_row
    assert "author\\_single\\_seed" not in seed_row
    assert "补充来源上下文" in report
    context_row = next(line for line in report.splitlines() if line.startswith("| protocol.seed |"))
    assert "作者公开说明 · author\\_single\\_seed" in context_row
    assert "| mse | 0.375 | [论文原文 · paper\\_table2" in report
    assert "| mae | 0.399 | [论文原文 · paper\\_table2" in report
    assert "选定论文实验数值复现通过" in report
    assert report_data == before


def test_reference_context_displays_exact_public_table_headers_and_selected_row(report_data):
    reader, sources = provenance_fixture(report_data)
    ref = {key: sources["paper_table2"][key] for key in ("source_id", "locator")}
    reader["reference_context"] = {
        "method": "DLinear", "dataset": "ETTh1", "features": "M", "input_length": 336,
        "forecast_horizon": 96, "scope": "selected_paper_experiment",
        "metric_columns": {"mse": "DLinear MSE", "mae": "DLinear MAE"},
        "table_headers": [{**ref, "quote": "Methods | DLinear* | DLinear*\nMetric | MSE | MAE"}],
        "table_row_evidence": [{**ref, "quote": "ETTh1 | 96 | 0.375 | 0.399"}],
        "derivation": "deterministic_selection_metadata"}
    report = build(report_data)
    assert "论文表格定位（本地确定性选择）" in report
    assert "| 数据集 | ETTh1 |" in report
    assert "| 输入长度 | 336 |" in report and "| 预测长度 | 96 |" in report
    assert "| MSE 列 | DLinear MSE |" in report and "| MAE 列 | DLinear MAE |" in report
    assert "> Methods \\| DLinear\\* \\| DLinear\\*" in report
    assert "> Metric \\| MSE \\| MAE" in report
    assert "> ETTh1 \\| 96 \\| 0.375 \\| 0.399" in report
    assert "本次运行证据另行记录" in report


def test_new_origin_links_require_exact_registered_locator(report_data):
    reader, _ = provenance_fixture(report_data)
    reader["protocol_provenance"]["protocol.seed"]["value_sources"][0]["locator"] = "run_longExp.py#L999"
    report = build(report_data)
    row = next(line for line in report.splitlines() if line.startswith("| seed |"))
    assert "run\\_longExp.py#L999（引用定位未匹配来源清单）" in row
    assert "](<https://github.com" not in row


@pytest.mark.parametrize("optional", [None, [], "legacy record without metadata"])
def test_missing_or_old_optional_metadata_does_not_invent_source_categories(report_data, optional):
    reader = report_data["repository_analysis"]["analyses"]["reader"]
    reader["protocol_provenance"] = optional
    reader["reference_context"] = optional
    report_data["repository_analysis"]["stages"][0]["rejection_diagnostics"] = optional
    report = build(report_data)
    assert "| 参数 | 抽取值 |" in report
    assert "数值依据" not in report
    assert "补充来源上下文" not in report
    assert "论文表格定位" not in report
    assert "作者代码设置或默认值" not in report
    assert "选定论文实验数值复现通过" in report


def test_documented_author_requirements_and_algorithm_constraint_are_not_runtime_evidence(report_data):
    builder = report_data["repository_analysis"]["analyses"]["builder"]
    builder.update(original_requirements_basis="documented_author_setup_not_verified_runtime",
                   changes_algorithm_role="constraint_on_unexecuted_proposals")
    report = build(report_data)
    assert "作者原始依赖**: numpy, torch==1.9.0" in report
    assert "原始依赖证据范围**: 作者声明的安装清单；未核实论文实验当时实际安装的环境。" in report
    assert "算法变更约束**: 兼容候选应保留冻结算法" in report
    assert "不提供运行核验证据" in report
    assert "兼容候选状态**: 未执行" in report
    assert "选定论文实验数值复现通过" in report

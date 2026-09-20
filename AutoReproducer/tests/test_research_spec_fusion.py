"""P1-⑦ 冻结 ResearchSpec + 隐藏 Holdout 验收（融合 ScholarAgent spec.py/coordinator.py）

覆盖:
1. ResearchSpec 校验（metric_key/direction/max_trials/holdout_repeats 边界）;
2. sha256 稳定性（notes/version 不入指纹; 契约字段变动即变化）;
3. improves / target_reached 判定（maximize/minimize + min_delta）;
4. from_dict/from_json/to_dict round-trip 与非法字段兜底;
5. verify_frozen_spec（声明 sha256 不一致 -> 拒绝）;
6. run_hidden_holdout（通过/不稳定/未达标/skipped/执行异常）;
7. OptimizerAgent 接入（兼容旧行为、spec 冻结校验、按 spec 判定 Keep、
   best 候选多轮 holdout、spec_pass 输出）。
"""
import pytest

from src.agents.optimizer import OptimizerAgent
from src.optimizer.research_spec import (
    InvalidResearchSpec,
    ResearchSpec,
    run_hidden_holdout,
    verify_frozen_spec,
)

_SPEC = {
    "metric_key": "accuracy",
    "direction": "maximize",
    "target_score": 0.9,
    "holdout_repeats": 3,
    "max_trials": 4,
}


class _ScriptedLLM:
    """返回固定 JSON suggestions 的假 LLM。"""

    def chat(self, prompt: str, task: str = "", **kwargs) -> str:
        return '{"suggestions": ["方向A", "方向B"]}'


class _FixedSim:
    """注入 OptimizerAgent 的确定性模拟器。"""

    def __init__(self, reward: float = 0.05):
        self._reward = reward

    def __call__(self, arm: str, baseline: float):
        return self._reward, {"type": "fixed", "arm": arm}


def _stable_runner(arm: str, index: int) -> dict:
    return {"exit_code": 0, "metrics": {"accuracy": 0.92}}


# ================================================================ 1. 校验

def test_spec_requires_metric_key():
    with pytest.raises(InvalidResearchSpec):
        ResearchSpec(metric_key="  ")


def test_spec_rejects_bad_direction():
    with pytest.raises(InvalidResearchSpec):
        ResearchSpec(metric_key="acc", direction="sideways")


def test_spec_clamps_trials_and_repeats():
    spec = ResearchSpec.from_dict(
        {"metric_key": "acc", "max_trials": 999, "holdout_repeats": 0})
    assert spec.max_trials == 20            # clamp 上限
    assert spec.holdout_repeats == 1        # clamp 下限


def test_spec_frozen_immutable():
    spec = ResearchSpec(metric_key="acc")
    with pytest.raises(AttributeError):     # frozen dataclass 不可修改
        spec.metric_key = "loss"


# ================================================================ 2. sha256

def test_sha256_stable_ignores_notes():
    a = ResearchSpec(metric_key="acc", notes="第一版")
    b = ResearchSpec(metric_key="acc", notes="第二版")
    assert a.sha256() == b.sha256()
    assert len(a.sha256()) == 64


def test_sha256_changes_with_contract_field():
    a = ResearchSpec(metric_key="acc", direction="maximize")
    b = ResearchSpec(metric_key="acc", direction="minimize")
    assert a.sha256() != b.sha256()

    c = ResearchSpec(metric_key="acc", target_score=0.9)
    d = ResearchSpec(metric_key="acc", target_score=0.95)
    assert c.sha256() != d.sha256()


# ================================================================ 3. 判定

def test_improves_maximize_with_min_delta():
    spec = ResearchSpec(metric_key="acc", direction="maximize",
                        min_delta=0.02)
    assert spec.improves(0.90, 0.85)        # +0.05 > 0.02
    assert not spec.improves(0.86, 0.85)    # +0.01 < 0.02
    assert not spec.improves(0.84, 0.85)    # 退步


def test_improves_minimize_with_min_delta():
    spec = ResearchSpec(metric_key="loss", direction="minimize",
                        min_delta=0.01)
    assert spec.improves(0.80, 0.85)        # -0.05 > 0.01
    assert not spec.improves(0.845, 0.85)   # -0.005 < 0.01


def test_target_reached():
    up = ResearchSpec(metric_key="acc", direction="maximize",
                      target_score=0.9)
    assert up.target_reached(0.91)
    assert not up.target_reached(0.89)

    down = ResearchSpec(metric_key="loss", direction="minimize",
                        target_score=0.1)
    assert down.target_reached(0.09)
    assert not down.target_reached(0.11)

    none = ResearchSpec(metric_key="acc", direction="maximize")
    assert none.target_reached(0.99) is False   # 未设目标永不达成


# ================================================================ 4. 序列化

def test_from_dict_round_trip_and_coercion():
    payload = {
        "metric_key": "acc", "direction": "maximize",
        "target_score": "0.9", "max_trials": "5",
        "editable_files": "model.py, utils.py",
        "protected_files": ["tests", "data"],
    }
    spec = ResearchSpec.from_dict(payload)
    assert spec.target_score == pytest.approx(0.9)
    assert spec.max_trials == 5
    assert spec.editable_files == ("model.py", "utils.py")

    restored = ResearchSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.sha256() == spec.sha256()


def test_from_json_and_invalid_input():
    spec = ResearchSpec.from_json('{"metric_key": "acc"}')
    assert spec.metric_key == "acc"
    with pytest.raises(InvalidResearchSpec):
        ResearchSpec.from_json("not json{")
    with pytest.raises(InvalidResearchSpec):
        ResearchSpec.from_dict([])          # 非 dict


# ================================================================ 5. 冻结校验

def test_verify_frozen_spec():
    spec = ResearchSpec.from_dict(_SPEC)
    ok, recomputed = verify_frozen_spec(spec)
    assert ok and recomputed == spec.sha256()

    bad, message = verify_frozen_spec(spec, declared_sha256="deadbeef")
    assert not bad and "hash mismatch" in message


# ================================================================ 6. Holdout

def test_holdout_passed_with_stable_runs():
    spec = ResearchSpec.from_dict(_SPEC)    # 3 轮, target 0.9
    out = run_hidden_holdout(spec, "arm-A", _stable_runner)
    assert out["mode"] == "hidden_holdout"
    assert out["candidate_arm"] == "arm-A"
    assert len(out["runs"]) == 3
    assert out["mean_score"] == pytest.approx(0.92)
    assert out["std_score"] == pytest.approx(0.0)
    assert out["target_ok"] is True and out["stable"] is True
    assert out["passed"] is True


def test_holdout_fails_on_unstable_runs():
    def flaky(arm, index):
        if index == 2:
            return {"exit_code": 1, "metrics": {}}
        return {"exit_code": 0, "metrics": {"accuracy": 0.92}}

    spec = ResearchSpec.from_dict(_SPEC)
    out = run_hidden_holdout(spec, "arm-A", flaky)
    assert out["stable"] is False
    assert out["passed"] is False


def test_holdout_fails_when_target_not_reached():
    def low(arm, index):
        return {"exit_code": 0, "metrics": {"accuracy": 0.85}}

    spec = ResearchSpec.from_dict(_SPEC)    # target 0.9
    out = run_hidden_holdout(spec, "arm-A", low)
    assert out["target_ok"] is False
    assert out["passed"] is False


def test_holdout_minimize_direction():
    spec = ResearchSpec(metric_key="loss", direction="minimize",
                        target_score=0.1, holdout_repeats=2)

    def runner(arm, index):
        return {"exit_code": 0, "metrics": {"loss": 0.09}}

    out = run_hidden_holdout(spec, "arm-A", runner)
    assert out["passed"] is True
    assert out["mean_score"] == pytest.approx(0.09)


def test_holdout_skipped_without_runner_or_candidate():
    spec = ResearchSpec(metric_key="acc")
    no_runner = run_hidden_holdout(spec, "arm-A", None)
    assert no_runner["mode"] == "skipped" and no_runner["passed"] is False

    no_candidate = run_hidden_holdout(spec, None, _stable_runner)
    assert no_candidate["mode"] == "skipped" and no_candidate["passed"] is False


def test_holdout_survives_runner_exception():
    def boom(arm, index):
        raise RuntimeError("executor killed")

    spec = ResearchSpec.from_dict(_SPEC)
    out = run_hidden_holdout(spec, "arm-A", boom)
    assert out["passed"] is False
    assert out["runs"][0]["exit_code"] == 1
    assert "error" in out["runs"][0]


# ================================================================ 7. Agent 接入

def _agent(spec=None, reward=0.05, runner=None,
           max_trials=4) -> OptimizerAgent:
    return OptimizerAgent(llm_client=_ScriptedLLM(), max_trials=max_trials,
                          simulator=_FixedSim(reward), spec=spec,
                          holdout_runner=runner)


def _input() -> dict:
    return {
        "paper_info": {"method": "DNN", "metrics": {"accuracy": 0.85}},
        "validation": {"is_reproduced": True},
    }


def test_agent_legacy_behavior_without_spec():
    """不传 spec 时行为与原实现一致（向后兼容）。"""
    agent = _agent()
    out = agent.run(_input())
    assert out["optimized"] is True
    assert out["budget_used"] <= out["budget"]
    assert out["spec_frozen"] is False and out["spec_sha256"] is None
    assert out["holdout"]["mode"] == "disabled"
    # 无 spec 时按相对 reward 判定 Keep
    assert all(r["kept"] is True for r in out["optimization_report"])


def test_agent_frozen_spec_happy_path():
    spec = ResearchSpec.from_dict(_SPEC)
    agent = _agent(spec=spec, reward=0.05, runner=_stable_runner)
    out = agent.run(_input())

    assert out["optimized"] is True
    assert out["spec_frozen"] is True
    assert out["spec_sha256"] == spec.sha256()
    # 隐藏 Holdout 对 best 候选执行 3 轮
    ho = out["holdout"]
    assert ho["mode"] == "hidden_holdout"
    assert ho["candidate_arm"] == out["best_arm"]
    assert ho["mean_score"] == pytest.approx(0.92)
    assert out["spec_pass"] is True
    # Keep 按契约判定（result = baseline*(1+reward) > baseline）
    assert all(r["kept"] is True for r in out["optimization_report"])


def test_agent_rejects_spec_on_hash_mismatch():
    tampered = dict(_SPEC, spec_sha256="tampered-hash")
    agent = _agent(spec=tampered)
    out = agent.run(_input())
    assert out["optimized"] is False
    assert out["spec_frozen"] is False
    assert "冻结" in out["reason"] or "hash" in out["reason"]


def test_agent_minimize_direction_keeps_negative_reward():
    """minimize 方向：reward 为负（结果下降）反而是 Keep。"""
    spec = ResearchSpec(metric_key="loss", direction="minimize",
                        target_score=0.1, holdout_repeats=2)
    agent = _agent(spec=spec, reward=-0.05)
    out = agent.run(_input())
    # reward -0.05 => result 0.8075 < baseline 0.85 => 契约判 Keep
    assert all(r["kept"] is True for r in out["optimization_report"])

    # 对照：无 spec 时负 reward 一律 Reject
    legacy = _agent(reward=-0.05).run(_input())
    assert all(r["kept"] is False for r in legacy["optimization_report"])


def test_agent_holdout_fail_marks_spec_pass_false():
    def flaky(arm, index):
        if index == 2:
            return {"exit_code": 1, "metrics": {}}
        return {"exit_code": 0, "metrics": {"accuracy": 0.92}}

    spec = ResearchSpec.from_dict(_SPEC)
    agent = _agent(spec=spec, reward=0.05, runner=flaky)
    out = agent.run(_input())
    assert out["holdout"]["passed"] is False
    assert out["spec_pass"] is False


def test_agent_spec_in_input_data_overrides_constructor():
    """input_data["spec"] 优先生效（构造参数兜底）。"""
    agent = _agent(spec=None, reward=0.05, runner=_stable_runner)
    out = agent.run({**_input(), "spec": _SPEC})
    assert out["spec_frozen"] is True
    assert out["spec_sha256"] == ResearchSpec.from_dict(_SPEC).sha256()
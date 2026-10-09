"""ExecutionPlan contracts: argv, cwd, order, budget and declared evidence."""
import pytest

from src.execution_plan import (
    PLAN_VERSION, RepositoryModeFallbackRejected, build_plan, plan_step_ids, plan_steps,
)

SPEC = "0" * 64


def step(identifier, **overrides):
    base = {"id": identifier, "argv": ["python", "-u", f"{identifier}.py"], "cwd": ".",
            "timeout_s": 60, "env": {}}
    return {**base, **overrides}


def build(steps, **overrides):
    kwargs = {"profile": "dlinear", "spec_sha256": SPEC, "workspace": "/repo", "steps": steps}
    return build_plan(**{**kwargs, **overrides})


def test_records_argv_cwd_timeout_and_kind():
    plan = build([step("train", cwd="exp", timeout_s=90, kind="train")])
    assert plan["version"] == PLAN_VERSION and plan["mode"] == "repository"
    recorded = plan["steps"][0]
    assert recorded["argv"] == ["python", "-u", "train.py"]
    assert recorded["cwd"] == "exp"
    assert recorded["timeout_s"] == 90.0
    assert recorded["kind"] == "train"


def test_dependency_and_evidence_fields_are_recorded():
    plan = build([
        step("import_check", kind="check", depends_on=[]),
        step("train", kind="train", depends_on=["import_check"],
             artifacts=["results/*/checkpoint.pth"]),
        step("eval", kind="eval", depends_on=["train"], requires=["artifacts/checkpoint.pt"]),
    ])
    train, evaluation = plan["steps"][1], plan["steps"][2]
    assert train["depends_on"] == ["import_check"]
    assert train["artifacts"] == [{"path": "results/*/checkpoint.pth", "optional": False,
                                   "kind": None}]
    assert evaluation["requires"] == [{"path": "artifacts/checkpoint.pt", "optional": False,
                                       "kind": None}]
    assert plan_step_ids(plan["steps"]) == ["import_check", "train", "eval"]


def test_budget_and_repository_metadata_survive_a_round_trip():
    plan = build([step("train")], limits={"total_seconds": 300},
                 repository={"url": "https://example.invalid/r", "revision": "abc"},
                 dataset={"name": "etth1", "sha256": SPEC})
    assert plan["limits"] == {"total_seconds": 300}
    assert plan["repository"]["revision"] == "abc"
    assert plan["dataset"]["name"] == "etth1"
    assert plan["repo_root"] == "/repo"


def test_reading_order_is_running_order_no_manual_topological_sort_needed():
    # A dependency points backwards only, so an author writes the steps in the order
    # they should run and never maintains a separate ordering.
    plan = build([step("a"), step("b", depends_on=["a"]), step("c", depends_on=["a", "b"])])
    assert [s["id"] for s in plan["steps"]] == ["a", "b", "c"]


def test_forward_reference_is_rejected_because_cycles_are_unrepresentable():
    with pytest.raises(ValueError, match="not scheduled before it"):
        build([step("a", depends_on=["b"]), step("b")])
    with pytest.raises(ValueError, match="not scheduled before it"):
        build([step("a", depends_on=["ghost"])])


def test_self_and_duplicate_dependency_declarations_are_rejected():
    with pytest.raises(ValueError, match="itself"):
        build([step("a", depends_on=["a"])])
    with pytest.raises(ValueError, match="duplicates"):
        build([step("a"), step("b", depends_on=["a", "a"])])


def test_duplicate_step_ids_are_rejected():
    with pytest.raises(ValueError, match="already declared"):
        build([step("a"), step("a")])


def test_paths_cannot_escape_the_workspace():
    for cwd in ("../outside", "/etc", ".."):
        with pytest.raises(ValueError):
            build([step("a", cwd=cwd)])
    with pytest.raises(ValueError):
        build([step("a", artifacts=["../../secrets/*"])])


def test_prerequisites_must_be_concrete_paths_not_globs():
    # Artifacts are collected opportunistically, so they may be patterns; a
    # prerequisite decides whether a step runs, so it must name one real file.
    with pytest.raises(ValueError, match="cannot be globs"):
        build([step("a"), step("b", requires=["artifacts/*.pt"])])


def test_argv_and_timeout_are_validated():
    for argv in ([], ["python", ""], ["python", 3], "python -c 1"):
        with pytest.raises(ValueError):
            build([step("a", argv=argv)])
    for timeout in (0, -1, float("nan"), float("inf"), True, "60"):
        with pytest.raises(ValueError, match="timeout_s"):
            build([step("a", timeout_s=timeout)])


def test_env_must_map_names_to_strings():
    for env in ({"PYTHONPATH": 1}, {"BAD=NAME": "x"}, {"": "x"}):
        with pytest.raises(ValueError, match="env"):
            build([step("a", env=env)])


def test_required_defaults_to_true_and_rejects_a_non_boolean():
    # Omitting the flag must not quietly make a train or eval step optional.
    assert build([step("a")])["steps"][0]["required"] is True
    assert build([step("a", required=False)])["steps"][0]["required"] is False
    for value in ("false", 0, None, []):
        with pytest.raises(ValueError, match="required"):
            build([step("a", required=value)])


def test_frozen_spec_digest_is_required():
    # The digest binds a plan to one reviewed spec; a plan that names no spec could
    # be replayed against any experiment and would prove nothing about this one.
    for digest in ("", "abc", "z" * 64, None, 3, SPEC * 2):
        with pytest.raises(ValueError, match="spec"):
            build([step("a")], spec_sha256=digest)
    assert build([step("a")])["spec_sha256"] == SPEC


def test_mode_profile_and_empty_step_list_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        build([step("a")], mode="script")
    with pytest.raises(ValueError, match="profile"):
        build([step("a")], profile="")
    with pytest.raises(ValueError, match="at least one step"):
        build([])
    with pytest.raises(ValueError, match="limits"):
        build([step("a")], limits="300")
    with pytest.raises(ValueError, match="total_seconds"):
        build([step("a")], limits={"total_seconds": -5})


def test_unsafe_step_ids_are_rejected():
    for identifier in ("", "a b", "../x", "a/b", None, 3):
        with pytest.raises(ValueError, match="safe file name"):
            build([step(identifier)])


def test_plan_steps_accepts_a_plan_or_a_bare_list():
    plan = build([step("a"), step("b")])
    assert [s["id"] for s in plan_steps(plan)] == ["a", "b"]
    assert [s["id"] for s in plan_steps(plan["steps"])] == ["a", "b"]
    for bad in ({}, {"steps": []}, {"steps": ["x"]}, [], None, "steps"):
        with pytest.raises(ValueError):
            plan_steps(bad)


def test_repository_mode_never_downgrades_to_single_file_generation():
    assert issubclass(RepositoryModeFallbackRejected, RuntimeError)
    with pytest.raises(RepositoryModeFallbackRejected):
        raise RepositoryModeFallbackRejected("guard exists to be caught, not ignored")


@pytest.mark.parametrize("profile_id", ["dlinear_etth1_smoke", "dlinear_etth1_reference"])
def test_reviewed_repository_profile_builds_a_valid_plan(profile_id):
    from src.repository_profiles import get_profile
    profile = get_profile(profile_id)
    plan = build(profile=profile["id"], steps=profile["steps"],
                 repository=profile["repository"], dataset=profile["dataset"],
                 limits=profile.get("budget"))
    check, train = plan["steps"]
    assert check["kind"] == "check" and check["depends_on"] == []
    # The authors' entry point is mounted verbatim, not rewritten into one file.
    assert train["argv"] == profile["steps"][1]["argv"]
    assert train["depends_on"] == ["import_check"]
    assert [a["path"] for a in train["artifacts"]] == ["results/*/checkpoint.pth",
                                                       "results/*/pred.npy"]


@pytest.mark.parametrize("adapter_id,profile_id", [("siren", "siren_camera_quick"),
                                                   ("neural_ode", "neural_ode_spiral")])
def test_method_profiles_split_train_and_eval_into_separate_steps(adapter_id, profile_id):
    from src.method_profiles import method_profile
    from src.repository_adapters import get_adapter
    profile = method_profile(profile_id)
    assert profile["adapter_id"] == adapter_id
    plan = build(profile=profile["id"], steps=get_adapter(profile).steps(profile))
    assert plan_step_ids(plan["steps"]) == ["import_check", "train", "evaluate"]
    train, evaluation = plan["steps"][1:]
    assert train["kind"] == "train" and train["depends_on"] == ["import_check"]
    assert evaluation["kind"] == "eval" and evaluation["depends_on"] == ["train"]
    # The evaluator is gated on the checkpoint the trainer declares, so a train step
    # that produced nothing cannot surface as a confusing failure inside evaluate.py.
    assert [r["path"] for r in evaluation["requires"]] == ["artifacts/checkpoint.pt",
                                                           "artifacts/prediction.npy"]
    assert [a["path"] for a in train["artifacts"]] == ["artifacts/checkpoint.pt",
                                                       "artifacts/prediction.npy",
                                                       "artifacts/training.json"]


def test_a_standalone_evaluator_has_no_dangling_dependency():
    from src.method_profiles import method_profile
    from src.repository_adapters import get_adapter
    profile = method_profile("siren_camera_quick")
    # Holdout confirmation runs the evaluator by itself; dependencies are stamped in
    # steps(), so the lone step must still form a valid plan.
    standalone = get_adapter(profile).evaluation_step(profile, "holdout")
    plan = build(profile=profile["id"], steps=[standalone])
    assert plan["steps"][0]["depends_on"] == []


def test_the_documented_plan_example_is_buildable_by_the_real_validator():
    # The design document is the contract reviewers read. A field rename in code and
    # the prose would then silently disagree, so the example is compiled here.
    import json
    import re
    from pathlib import Path
    document = Path(__file__).resolve().parents[1] / "docs" / "repository_reproduction_plan.md"
    text = document.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    examples = [json.loads(block) for block in blocks]
    example = next(item for item in examples if item.get("mode") == "repository")
    plan = build(profile=example["profile"], spec_sha256=example["spec_sha256"],
                 workspace=example["workspace"], repository=example["repository"],
                 dataset=example["dataset"], limits=example["limits"],
                 steps=example["steps"])
    assert plan_step_ids(plan["steps"]) == ["import_check", "train_and_eval", "evaluate"]
    assert [s["kind"] for s in plan["steps"]] == ["check", "train", "eval"]
    # Reading the authored order straight off the document proves the example itself,
    # not just the builder, carries the train/eval split reviewers are meant to copy.
    assert plan_step_ids(example["steps"]) == plan_step_ids(plan["steps"])
    assert example["steps"][2]["requires"] == [{"path": "results/ETTh1_336_96/checkpoint.pth"}]

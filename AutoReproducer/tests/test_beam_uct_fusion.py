"""P1-⑥ BeamUCT 双层搜索（方向级 UCB + 参数树 UCT + Beam top-k + 先验播种）"""
import pytest

from src.experience.experience_store import ExperienceStore
from src.optimizer.beam_uct import (
    Beam,
    BeamUCTSearch,
    BeamUCTSearchEngine,
    DirectionUCB,
    UCTSearch,
    direction_priors,
    params_key,
    seed_from_store,
    seed_from_summary,
    select_prior_candidate,
)
from src.optimizer.ucb_scheduler import UCBScheduler

_SPACE = {"lr": [1e-3, 1e-4], "bs": [32, 64, 128]}


def _validated(record):
    record["validated"] = record.get("validated", True)
    return record


# ---------------------------------------------------------------- UCTSearch

def test_uct_propose_exhausts_all_combinations():
    tree = UCTSearch(_SPACE)
    proposed = set()
    while not tree.exhausted():
        params = tree.propose()
        assert params is not None
        assert set(params) == set(_SPACE)
        key = params_key(params)
        assert key not in proposed          # 不重复
        proposed.add(key)
    assert tree.propose() is None           # 穷尽后返回 None
    assert len(proposed) == 2 * 3


def test_uct_report_back_propagates():
    tree = UCTSearch(_SPACE)
    params = tree.propose()
    assert params is not None
    tree.report(params, 1.0)
    assert tree.root.visits == 1
    assert tree.root.total_reward == 1.0
    # 路径上的每一层都更新
    assert all(c.visits == 1 for c in tree.root.children)
    # 第二次 propose 后回传另一组合
    params2 = tree.propose()
    assert params2 is not None
    assert params_key(params2) != params_key(params)
    tree.report(params2, 0.5)
    assert tree.root.visits == 2


def test_uct_minimize_direction_flips_reward():
    tree = UCTSearch({"x": [1, 2]}, direction="minimize")
    params = tree.propose()
    assert params is not None
    tree.report(params, 5.0)
    assert tree.root.total_reward == -5.0   # minimize 取负号


def test_uct_rejects_bad_space():
    with pytest.raises(ValueError):
        UCTSearch({})
    with pytest.raises(ValueError):
        UCTSearch({"a": list(range(9))})    # 单参数取值超 8
    with pytest.raises(ValueError):
        UCTSearch({f"p{i}": [1] for i in range(9)})   # 参数数超 8


# ---------------------------------------------------------------- Beam

def test_beam_keeps_top_width():
    beam = Beam(width=2)
    assert beam.offer({"a": 1}, 0.3) is True
    assert beam.offer({"a": 2}, 0.9) is True
    assert beam.offer({"a": 3}, 0.5) is True
    assert len(beam.items()) == 2
    assert beam.best() == {"a": 2}          # 最高分保留
    assert beam.best_score({"a": 3}) == 0.5
    assert beam.best_score({"a": 9}) is None


def test_beam_minimize():
    beam = Beam(width=2, direction="minimize")
    beam.offer({"a": 1}, 5.0)
    beam.offer({"a": 2}, 1.0)
    beam.offer({"a": 3}, 3.0)
    assert beam.best() == {"a": 2}


def test_beam_rejects_zero_width():
    with pytest.raises(ValueError):
        Beam(width=0)


# ---------------------------------------------------------------- DirectionUCB

def test_direction_ucb_unexplored_first_then_formula():
    ucb = DirectionUCB(["a", "b", "c"])
    picked = []
    for _ in range(3):
        arm = ucb.select()
        picked.append(arm)
        ucb.update(arm, 0.5)                # 已选即更新, 推进强制探索
    assert set(picked) == {"a", "b", "c"}   # 未尝试方向逐个被强制探索
    # 全部拉过一次后, UCB1 选择应有探索项（不抛错、在臂内）
    for _ in range(10):
        arm = ucb.select()
        assert arm in {"a", "b", "c"}


def test_direction_ucb_seed_biases_selection():
    ucb = DirectionUCB(["a", "b"])
    ucb.update("a", 0.1)                    # 真实低分
    ucb.seed("b", count=5, mean=0.9)        # 历史高分先验
    picked = [ucb.select() for _ in range(5)]
    assert picked.count("b") > picked.count("a")


def test_direction_ucb_retire():
    ucb = DirectionUCB(["a", "b"])
    ucb.update("a", 0.5)
    ucb.update("b", 0.5)
    ucb.retire("a")
    assert ucb.retired() == ["a"]
    assert ucb.select() == "b"
    ucb.retire("b")
    with pytest.raises(ValueError):
        ucb.select()


def test_direction_ucb_update_mean():
    ucb = DirectionUCB(["a"])
    ucb.update("a", 0.0)
    ucb.update("a", 1.0)
    assert ucb.means()["a"] == pytest.approx(0.5)
    assert ucb.counts()["a"] == 2
    assert ucb.total_pulls == 2


def test_direction_ucb_rejects_empty():
    with pytest.raises(ValueError):
        DirectionUCB([])


# ---------------------------------------------------------------- BeamUCTSearch

def test_beam_uct_facade():
    search = BeamUCTSearch(_SPACE, beam_width=2)
    seen = set()
    while not search.exhausted():
        params = search.propose()
        assert params is not None
        score = 0.5 if params["lr"] == 1e-4 else 0.3
        search.report(params, score)
        seen.add(params_key(params))
    assert len(seen) == 6
    best = search.best()
    assert best is not None and best["lr"] == 1e-4


# ---------------------------------------------------------------- Engine

def test_engine_full_loop_and_best():
    spaces = {"arch": _SPACE, "data": {"k": [1, 2]}}
    engine = BeamUCTSearchEngine(list(spaces), spaces, beam_width=2)
    proposals = 0
    while not engine.exhausted():
        pick = engine.propose()
        assert pick is not None
        direction, params = pick
        assert direction in spaces
        score = 1.0 if params.get("lr") == 1e-4 else 0.2
        engine.report(direction, params, score)
        proposals += 1
    assert proposals == 6 + 2               # arch:2*3, data:2
    best = engine.best()
    assert best is not None
    assert best[2] == 1.0
    assert best[1]["lr"] == 1e-4


def test_engine_retires_exhausted_direction():
    spaces = {"fast": {"x": [1]}, "wide": {"y": [1, 2]}}
    engine = BeamUCTSearchEngine(["fast", "wide"], spaces)
    combos = set()
    while True:
        pick = engine.propose()
        if pick is None:
            break
        engine.report(pick[0], pick[1], 0.1)
        combos.add(pick[0] + params_key(pick[1]))
    # 拉完 fast（仅 1 组合）后应自动退役并转向 wide
    assert combos == {"fast" + params_key({"x": 1}),
                      "wide" + params_key({"y": 1}),
                      "wide" + params_key({"y": 2})}
    assert "fast" in engine.ucb.retired()


def test_engine_params_in_space():
    engine = BeamUCTSearchEngine(["arch"], {"arch": _SPACE})
    assert engine.params_in_space("arch", {"lr": 1e-4, "bs": 64}) is True
    assert engine.params_in_space("arch", {"lr": 9.9, "bs": 64}) is False
    assert engine.params_in_space("arch", {"lr": 1e-4}) is False
    assert engine.params_in_space("nope", {"lr": 1e-4, "bs": 64}) is False


# ---------------------------------------------------------------- 先验播种

def test_direction_priors():
    experiences = [
        _validated({"direction": "arch", "score": 0.7}),
        _validated({"direction": "arch", "score": 0.9}),
        _validated({"direction": "arch", "score": 0.8}),
        _validated({"direction": "data", "score": 0.4}),
        _validated({"direction": "data", "score": 0.5}),
        {"direction": "arch", "score": 0.99, "validated": False},  # 未验证忽略
    ]
    priors = direction_priors(experiences, ["arch", "data"])
    count_arch, mean_arch = priors["arch"]
    assert count_arch == 3                  # 伪拉取 = min(数, 3)
    assert mean_arch == pytest.approx((0.9 + 0.8 + 0.7) / 3)  # top-3 均值
    assert priors["data"][1] == pytest.approx((0.5 + 0.4) / 2)


def test_seed_from_summary():
    ucb = DirectionUCB(["arch", "data"])
    summary = {"directions": {
        "arch": {"count": 10, "mean": 0.8},
        "data": {"count": 2, "mean": 0.3},
        "other": {"count": 9, "mean": 0.6},
    }}
    priors = seed_from_summary(summary, ucb, directions=["arch", "data"])
    assert set(priors) == {"arch", "data"}
    assert ucb.counts()["arch"] == 3        # 伪拉取封顶 3
    assert ucb.means()["arch"] == pytest.approx(0.8)
    assert "other" not in ucb.arms          # 未在 directions 内不播种


def test_seed_from_store(tmp_path):
    store = ExperienceStore(tmp_path / "exp.jsonl")
    store.append(_validated({"repo": "r1", "direction": "arch",
                             "score": 0.6, "params": {}}))
    store.append({"repo": "r1", "direction": "arch",
                  "score": 0.99, "validated": False})   # 不参与
    ucb = DirectionUCB(["arch", "data"])
    priors = seed_from_store(store, repo="r1", ucb=ucb)
    assert priors["arch"][1] == pytest.approx(0.6)
    assert "data" not in priors
    assert ucb.means()["arch"] == pytest.approx(0.6)


def test_select_prior_candidate():
    spaces = {"arch": _SPACE}
    experiences = [
        _validated({"direction": "arch", "score": 0.5, "params": {"lr": 1e-3, "bs": 32}}),
        _validated({"direction": "arch", "score": 0.9, "params": {"lr": 1e-4, "bs": 64}}),
        _validated({"direction": "arch", "score": 0.95, "params": {"lr": 1e-4, "bs": 64}}),  # 同组合更高
        _validated({"direction": "arch", "score": 0.8, "params": {"lr": 1e-4, "bs": 999}}),  # 空间外
    ]
    prior = select_prior_candidate(experiences, spaces)
    assert prior is not None
    assert prior["direction"] == "arch"
    assert prior["params"] == {"lr": 1e-4, "bs": 64}
    assert prior["prior_score"] == pytest.approx(0.925)  # (0.9+0.95)/2
    assert prior["matched"] == 2


def test_select_prior_candidate_none_when_out_of_space():
    experiences = [_validated(
        {"direction": "arch", "score": 0.9, "params": {"lr": 1e-4, "bs": 999}})]
    assert select_prior_candidate(experiences, {"arch": _SPACE}) is None
    # 方向不在空间中
    experiences2 = [_validated(
        {"direction": "zzz", "score": 0.9, "params": {"lr": 1e-4, "bs": 64}})]
    assert select_prior_candidate(experiences2, {"arch": _SPACE}) is None


# ---------------------------------------------------------------- UCBScheduler 融合

def test_ucb_scheduler_seed():
    sched = UCBScheduler(["a", "b"], budget=10)
    sched.seed("a", count=5, mean=0.9)
    assert sched.arms["a"]["n"] == 5
    assert sched.arms["a"]["Q"] == 0.9
    assert sched.total_pulls == 0            # 先验不消耗本轮预算
    # b 未拉过仍是第一个被强制探索的
    assert sched.select_arm() == "b"


def test_ucb_scheduler_retire():
    sched = UCBScheduler(["a", "b"], budget=10)
    sched.seed("a", count=1, mean=1.0)
    sched.update("b", 0.0)
    sched.retire("a")
    assert sched.retired() == ["a"]
    assert {sched.select_arm() for _ in range(3)} == {"b"}


def test_ucb_scheduler_all_retired():
    sched = UCBScheduler(["a"], budget=10)
    sched.retire("a")
    assert sched.select_arm() is None
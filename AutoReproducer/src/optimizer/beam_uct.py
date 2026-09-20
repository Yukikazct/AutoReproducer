"""BeamUCT - 双层搜索调度器（方向级 UCB + 参数树 UCT + Beam top-k）。

融合自 ScholarAgent backend/app/research/optimizer/（MIT License）:
- direction_priors / select_candidate（policy.py）: 历史经验 -> 方向先验与首航候选；
- UCBSelector（ucb.py）: 方向级 UCB1, 支持 seed（伪拉取先验）与 retire（退役）;
- UCTSearch / Beam / BeamUCTSearch（beam_uct.py）: 参数树 UCT 展开 + top-k Beam。

与 AutoReproducer 已有 UCBScheduler（src/optimizer/ucb_scheduler.py）的分工:
- UCBScheduler: 预算感知的轻量方向调度, 已补充 seed/retire 能力;
- 本模块: 完整双层搜索 —— 方向级 UCB 选方向, 方向内参数树 UCT 展开完整参数组合,
  全局 Beam 保留历史 top-k, 并可被 ExperienceStore 的先验播种开局。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 参数空间规模上限（防搜索爆炸, 与 ScholarAgent 一致）
MAX_SPACE_PARAMS = 8
MAX_VALUES_PER_PARAM = 8

# 先验播种：每个方向最多注入的伪拉取次数
PRIOR_PSEUDO_PULLS = 3


def params_key(params: Dict[str, Any]) -> str:
    """参数组合的规范化字符串键（order 无关, 可哈希去重）。"""
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


# ---------------------------------------------------------------- 参数树 UCT

@dataclass
class UCTNode:
    """参数树节点：depth 层对应空间第 depth 个参数, 该节点分支于 param_key。"""
    depth: int
    params: Dict[str, Any]                 # 已确定的部分参数
    param_key: str                         # 本节点子节点按哪个空间键分支
    visits: int = 0
    total_reward: float = 0.0
    children: List["UCTNode"] = field(default_factory=list)
    _untried: List[Any] = field(default_factory=list)   # 尚未尝试的参数取值

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0


class UCTSearch:
    """参数树 UCT：每次 propose 扩展一个新全深度叶子（完整参数组合）。

    每层对应一个空间参数；descent 按 UCT 分数选分支，expansion 取该层下一个
    未试值；report() 把评估结果沿路径回传, 让后续 propose 偏向高分子树。
    """

    def __init__(
        self,
        space: Dict[str, List[Any]],
        exploration: float = 1.414,
        direction: str = "maximize",
    ) -> None:
        if not space:
            raise ValueError("UCTSearch needs a non-empty parameter space")
        if len(space) > MAX_SPACE_PARAMS:
            raise ValueError(
                f"parameter space too large: {len(space)} > {MAX_SPACE_PARAMS}")
        for key, values in space.items():
            if not values or len(values) > MAX_VALUES_PER_PARAM:
                raise ValueError(
                    f"parameter {key!r} needs 1..{MAX_VALUES_PER_PARAM} values")
        self.space_keys = list(space.keys())
        self.space = {key: list(values) for key, values in space.items()}
        self.exploration = exploration
        self.sign = 1.0 if direction == "maximize" else -1.0
        self.root = UCTNode(
            depth=0, params={}, param_key=self.space_keys[0],
            _untried=list(self.space[self.space_keys[0]]))

    # ------------------------------------------------------------ 生成

    def propose(self) -> Optional[Dict[str, Any]]:
        """展开一个新全深度叶子; 空间全遍历后返回 None。"""
        branch = self._find_branch(self.root)
        if branch is None:
            return None
        return self._complete(branch)

    def _find_branch(self, node: UCTNode) -> Optional[UCTNode]:
        if node._untried:
            return node
        ordered = sorted(
            node.children, key=lambda c: self._uct_score(node, c), reverse=True)
        for child in ordered:
            found = self._find_branch(child)
            if found is not None:
                return found
        return None

    def _make_child(self, node: UCTNode, value: Any) -> UCTNode:
        child_depth = node.depth + 1
        next_key = (self.space_keys[child_depth]
                    if child_depth < len(self.space_keys) else "")
        return UCTNode(
            depth=child_depth,
            params={**node.params, node.param_key: value},
            param_key=next_key,
            _untried=list(self.space[next_key]) if next_key else [],
        )

    def _complete(self, node: UCTNode) -> Dict[str, Any]:
        current = node
        while current.param_key:
            if current._untried:
                value = current._untried.pop(0)
                child = self._make_child(current, value)
                current.children.append(child)
                current = child
            else:
                next_node = self._best_child(current)
                if next_node is None:       # 理论不可达：param_key 非空必有子节点
                    break
                current = next_node
        return dict(current.params)

    def _uct_score(self, parent: UCTNode, child: UCTNode) -> float:
        exploration = self.exploration * math.sqrt(
            2 * math.log(max(parent.visits, 1)) / max(child.visits, 1))
        return child.mean_reward + exploration

    def _best_child(self, node: UCTNode) -> Optional[UCTNode]:
        best, best_score = None, float("-inf")
        for child in node.children:
            score = self._uct_score(node, child)
            if score > best_score:
                best, best_score = child, score
        return best

    # ------------------------------------------------------------ 回传

    def report(self, params: Dict[str, Any], score: float) -> None:
        """沿树路径回传一次评估结果（reward 按 direction 符号化）。"""
        node = self.root
        reward = self.sign * float(score)
        node.visits += 1
        node.total_reward += reward
        for key in self.space_keys:
            value = params.get(key)
            child = next(
                (c for c in node.children if c.params.get(key) == value), None)
            if child is None:
                break
            child.visits += 1
            child.total_reward += reward
            node = child

    def exhausted(self) -> bool:
        """树中是否已无任何未试取值。"""
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node._untried:
                return False
            stack.extend(node.children)
        return True


# ---------------------------------------------------------------- Beam top-k

class Beam:
    """按已验证分数保留 top-w 配置（方向内）。"""

    def __init__(self, width: int = 3, direction: str = "maximize") -> None:
        if width < 1:
            raise ValueError("beam width must be >= 1")
        self.width = width
        self.sign = 1.0 if direction == "maximize" else -1.0
        self._items: List[Tuple[float, Dict[str, Any]]] = []
        self._scores: Dict[str, float] = {}   # params_key -> 原始 score

    def offer(self, params: Dict[str, Any], score: float) -> bool:
        """记录一个已评估配置; 进入 beam 返回 True。"""
        entry = (self.sign * float(score), dict(params))
        self._items.append(entry)
        self._scores[params_key(params)] = float(score)
        self._items.sort(key=lambda item: item[0], reverse=True)
        self._items = self._items[: self.width]
        return any(
            params_key(item[1]) == params_key(params) for item in self._items)

    def best(self) -> Optional[Dict[str, Any]]:
        return dict(self._items[0][1]) if self._items else None

    def best_score(self, params: Dict[str, Any]) -> Optional[float]:
        """返回某组参数（若仍在 beam 内）的原始分数。"""
        return self._scores.get(params_key(params))

    def items(self) -> List[Dict[str, Any]]:
        return [dict(params) for _, params in self._items]


# ---------------------------------------------------------------- 方向级 UCB

class DirectionUCB:
    """方向级 UCB1 选择器（吸收 ScholarAgent UCBSelector 的 seed/retire）。

    - seed(): 注入历史先验（伪拉取次数 + 先验均值）, 搜索开局偏向已验证方向;
    - retire(): 退役方向（如该方向参数树已穷尽）, 后续不再选择。
    """

    def __init__(self, directions: List[str], exploration: float = 1.414) -> None:
        if not directions:
            raise ValueError("DirectionUCB needs at least one direction")
        self.exploration = exploration
        self._arms = list(dict.fromkeys(directions))
        self._counts: Dict[str, int] = {a: 0 for a in self._arms}
        self._means: Dict[str, float] = {a: 0.0 for a in self._arms}
        self._retired: set = set()
        self._total = 0

    @property
    def arms(self) -> List[str]:
        return list(self._arms)

    @property
    def total_pulls(self) -> int:
        return self._total

    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def means(self) -> Dict[str, float]:
        return dict(self._means)

    def select(self) -> str:
        """挑下一个方向; 未尝试方向优先, 其余按 UCB1 argmax。"""
        for arm in self._arms:
            if arm not in self._retired and self._counts[arm] == 0:
                return arm
        best_arm, best_value = None, float("-inf")
        for arm in self._arms:
            if arm in self._retired:
                continue
            value = self._means[arm] + self.exploration * math.sqrt(
                2 * math.log(self._total) / self._counts[arm])
            if value > best_value:
                best_arm, best_value = arm, value
        if best_arm is None:
            raise ValueError("all UCB directions are retired")
        return best_arm

    def seed(self, arm: str, count: int, mean: float) -> None:
        """注入历史先验（伪拉取次数 + 先验均值）, 不消耗真实预算。"""
        if arm not in self._counts:
            self._arms.append(arm)
            self._counts[arm] = 0
            self._means[arm] = 0.0
        self._counts[arm] = max(self._counts[arm], int(count))
        self._means[arm] = float(mean)

    def retire(self, arm: str) -> None:
        """退役一个方向（如参数树穷尽）, 不再参与选择。"""
        self._retired.add(arm)

    def retired(self) -> List[str]:
        return sorted(self._retired)

    def update(self, arm: str, score: float) -> None:
        """记录一次真实评估结果（增量均值）。"""
        if arm not in self._counts:
            self._arms.append(arm)
            self._counts[arm] = 0
            self._means[arm] = 0.0
        count, mean = self._counts[arm], self._means[arm]
        self._means[arm] = (mean * count + score) / (count + 1)
        self._counts[arm] = count + 1
        self._total += 1


# ---------------------------------------------------------------- 双层搜索

class BeamUCTSearch:
    """Facade: 绑定全局 beam 到每次 propose 的 UCT 展开。"""

    def __init__(
        self,
        space: Dict[str, List[Any]],
        beam_width: int = 3,
        exploration: float = 1.414,
        direction: str = "maximize",
    ) -> None:
        self.beam = Beam(width=beam_width, direction=direction)
        self.tree = UCTSearch(
            space=space, exploration=exploration, direction=direction)
        self.direction = direction

    def propose(self) -> Optional[Dict[str, Any]]:
        return self.tree.propose()

    def report(self, params: Dict[str, Any], score: float) -> None:
        self.tree.report(params, score)
        self.beam.offer(params, score)

    def best(self) -> Optional[Dict[str, Any]]:
        return self.beam.best()

    def exhausted(self) -> bool:
        return self.tree.exhausted()


class BeamUCTSearchEngine:
    """双层闭环: 方向级 UCB 选方向 -> 方向内 UCT 展开 -> Beam 保留 top-k。

    每个方向一棵参数树 + 一个 beam; 方向全部退役（各自的树穷尽）时视为
    整体搜索结束。支持 ExperienceStore 先验播种（seed_from_summary /
    seed_from_store / select_prior_candidate）。
    """

    def __init__(
        self,
        directions: List[str],
        spaces: Dict[str, Dict[str, List[Any]]],
        beam_width: int = 3,
        exploration: float = 1.414,
        direction: str = "maximize",
        ucb: Optional[DirectionUCB] = None,
    ) -> None:
        if not directions:
            raise ValueError("engine needs at least one direction")
        missing = [d for d in directions if d not in spaces]
        if missing:
            raise ValueError(f"missing param space for directions: {missing}")
        self.direction_mode = direction
        self.ucb = ucb or DirectionUCB(directions, exploration=exploration)
        self.trees: Dict[str, BeamUCTSearch] = {}
        for d in directions:
            self.trees[d] = BeamUCTSearch(
                space=spaces[d], beam_width=beam_width,
                exploration=exploration, direction=direction)

    # ------------------------------------------------------------ 主循环

    def propose(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """选方向并展开一个新参数组合; 全搜索空间穷尽返回 None。"""
        while True:
            try:
                arm = self.ucb.select()
            except ValueError:
                return None                     # 全部方向已退役
            tree = self.trees[arm]
            params = tree.propose()
            if params is not None:
                return arm, params
            self.ucb.retire(arm)                # 该方向树已穷尽, 退役换方向

    def report(self, direction: str, params: Dict[str, Any],
               score: float) -> None:
        """回传一次真实评估: 更新方向 UCB 与该方向树/beam。"""
        self.ucb.update(direction, score)
        self.trees[direction].report(params, score)

    # ------------------------------------------------------------ 查询

    def best(self) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """全局最优: 跨方向扫描各 beam 的最优分数。"""
        best_entry: Optional[Tuple[str, Dict[str, Any], float]] = None
        for arm, tree in self.trees.items():
            params = tree.best()
            if params is None:
                continue
            score = tree.beam.best_score(params)
            if score is None:
                continue
            if best_entry is None or score > best_entry[2]:
                best_entry = (arm, params, score)
        return best_entry

    def exhausted(self) -> bool:
        return all(t.exhausted() for t in self.trees.values())

    def params_in_space(self, direction: str,
                        params: Dict[str, Any]) -> bool:
        """校验一个候选参数组合是否落在该方向树的空间内（供播种过滤）。"""
        search = self.trees.get(direction)
        if search is None:
            return False
        tree = search.tree
        if set(params.keys()) != set(tree.space_keys):
            return False
        return all(params[k] in tree.space[k] for k in tree.space_keys)


# ---------------------------------------------------------------- 先验播种

def _top_k_mean(values: List[float], limit: int = PRIOR_PSEUDO_PULLS) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)[:limit]
    return sum(ordered) / len(ordered)


def direction_priors(
    experiences: List[Dict[str, Any]],
    directions: List[str],
) -> Dict[str, Tuple[int, float]]:
    """由验证过的经验记录, 计算每个方向的 (伪拉取次数, 先验均值)。

    伪拉取次数 = min(带分记录数, PRIOR_PSEUDO_PULLS), 防止单一方向先验过强;
    先验均值 = 该方向 top-k 得分的均值（缓解离群值影响）。
    """
    priors: Dict[str, Tuple[int, float]] = {}
    for direction in directions:
        scores = [
            float(r["score"])
            for r in experiences
            if r.get("validated")
            and str(r.get("direction", "")) == direction
            and isinstance(r.get("score"), (int, float))
        ]
        if scores:
            priors[direction] = (
                min(len(scores), PRIOR_PSEUDO_PULLS), _top_k_mean(scores))
    return priors


def seed_from_summary(
    summary: Dict[str, Any],
    ucb: DirectionUCB,
    directions: Optional[List[str]] = None,
) -> Dict[str, Tuple[int, float]]:
    """用 ExperienceStore.summarize() 的结构播方向先验（不依赖原始记录）。

    summary 形如 {"directions": {name: {"count": n, "mean": m}}}。
    """
    wanted = directions or list(summary.get("directions", {}))
    priors: Dict[str, Tuple[int, float]] = {}
    for name in wanted:
        entry = summary.get("directions", {}).get(name)
        if not entry or entry.get("mean") is None:
            continue
        count = min(int(entry.get("count", 0)), PRIOR_PSEUDO_PULLS)
        priors[name] = (count, float(entry["mean"]))
        ucb.seed(name, count, float(entry["mean"]))
    return priors


def seed_from_store(
    store: Any,
    repo: str,
    ucb: DirectionUCB,
    directions: Optional[List[str]] = None,
) -> Dict[str, Tuple[int, float]]:
    """从 ExperienceStore 实例播种方向先验（读取 validated 记录）。"""
    experiences = store.all(repo=repo, validated_only=True)
    priors = direction_priors(experiences, directions or list(ucb.arms))
    for arm, (count, mean) in priors.items():
        ucb.seed(arm, count, mean)
    return priors


def select_prior_candidate(
    experiences: List[Dict[str, Any]],
    spaces: Dict[str, Dict[str, List[Any]]],
) -> Optional[Dict[str, Any]]:
    """挑历史最优且仍在参数空间内的候选, 作为首航播种候选。

    返回 {"direction", "params", "prior_score", "matched"} 或 None。
    候选必须先落空间校验: 只在方向存在、集合完全一致且值合法时返回。
    """
    scored_groups: Dict[str, Dict[str, Any]] = {}
    for record in experiences:
        if not record.get("validated"):
            continue
        if not isinstance(record.get("score"), (int, float)):
            continue
        direction = str(record.get("direction", ""))
        params = record.get("params") or {}
        if not isinstance(params, dict) or not params:
            continue
        key = direction + "|" + params_key(params)
        group = scored_groups.setdefault(
            key, {"direction": direction, "params": params, "scores": []})
        group["scores"].append(float(record["score"]))

    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for group in scored_groups.values():
        direction = group["direction"]
        if direction not in spaces:
            continue
        params = group["params"]
        tree = spaces[direction]
        if set(params.keys()) != set(tree.keys()):
            continue
        if not all(params[k] in tree[k] for k in tree):
            continue
        prior = _top_k_mean(group["scores"])
        if prior > best_score:
            best = {
                "direction": direction,
                "params": dict(params),
                "prior_score": round(prior, 6),
                "matched": len(group["scores"]),
            }
            best_score = prior
    return best
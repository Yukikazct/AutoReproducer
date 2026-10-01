"""P1-⑨ 确定性仓库发现链测试。

覆盖：纯函数（URL 归一/提取/查询构造/curated/信任判定/评分/模式决策）、
discover_repositories 四层降级链（monkeypatch 网络层）、
ResourceFinderAgent 融合（确定性优先 + pin revision + repro_mode）、
resource_manager.fetch_code revision pin + 溯源标记写入。
全部避免真实网络。
"""
import json
import subprocess

import pytest

from src.agents import repo_discovery as rd
from src.agents.repo_discovery import (
    RepoCandidate,
    build_repo_discovery_query,
    build_github_fallback_queries,
    curated_repo_fallback_candidates,
    decide_reproduction_mode,
    discover_repositories,
    extract_plain_github_urls,
    github_repo_search_score,
    is_trusted_repo_candidate,
    normalize_github_repo_url,
    significant_tokens,
)
from src.agents.resource_finder import ResourceFinderAgent
from src.resource_manager import ResourceManager


# ---------------- 纯函数 ----------------

class TestNormalizeUrl:
    def test_standard(self):
        assert (normalize_github_repo_url(
            "https://github.com/a/b.git") ==
            "https://github.com/a/b")
        assert (normalize_github_repo_url(
            "  https://github.com/owner/repo  ") ==
            "https://github.com/owner/repo")

    def test_bare_owner_repo(self):
        assert (normalize_github_repo_url("owner/repo-name") ==
                "https://github.com/owner/repo-name")

    def test_invalid(self):
        assert normalize_github_repo_url("") == ""
        assert normalize_github_repo_url("https://example.com/x/y") == ""
        assert normalize_github_repo_url("随便写的文本") == ""


class TestExtractPlainUrls:
    def test_dedupe_and_clean(self):
        text = ("see https://github.com/a/b and https://github.com/a/b.git "
                "plus https://github.com/c/d/issues/1")
        urls = extract_plain_github_urls(text)
        assert urls == ["https://github.com/a/b",
                        "https://github.com/c/d"]

    def test_empty(self):
        assert extract_plain_github_urls("") == []
        assert extract_plain_github_urls("no urls here") == []


class TestBuildQuery:
    def test_arxiv_preferred(self):
        q = build_repo_discovery_query(
            {"arxiv_id": "2106.09685", "title": "Foo bar", "method": "m"})
        assert q == "2106.09685"

    def test_title_then_method(self):
        assert build_repo_discovery_query(
            {"title": "Attention Is All You Need"}) == \
            "Attention Is All You Need"
        assert build_repo_discovery_query({"method": "PINN"}) == "PINN"

    def test_empty(self):
        assert build_repo_discovery_query({}) == ""


class TestCuratedFallback:
    def test_builtin_pinn(self):
        cands = curated_repo_fallback_candidates("pinn driven by physics")
        assert any("PINNs" in c.repo_urls[0] for c in cands)

    def test_transformer(self):
        cands = curated_repo_fallback_candidates(
            "Attention is all you need revisited")
        assert any("annotated-transformer" in c.repo_urls[0]
                   for c in cands)

    def test_env_extension(self, monkeypatch):
        monkeypatch.setenv(
            "AUTOREPRO_CURATED_REPOS",
            json.dumps({"我的论文": [{"repo_url": "https://github.com/me/wa",
                                      "repo_name": "me/wa",
                                      "description": "my paper repo"}]}))
        cands = curated_repo_fallback_candidates("关于我的论文的方法")
        assert any(c.repo_urls[0] == "https://github.com/me/wa"
                   for c in cands)

    def test_no_match(self):
        assert curated_repo_fallback_candidates(
            "quantum entropy reproduction") == []


class TestTrustedCandidate:
    def test_pinned_sources_always_trusted(self):
        assert is_trusted_repo_candidate(
            "anything", RepoCandidate(source="user_preference",
                                      repo_urls=["https://github.com/a/b"]))
        assert is_trusted_repo_candidate(
            "anything", RepoCandidate(source="curated_fallback",
                                      repo_urls=["https://github.com/a/b"]))

    def test_token_overlap(self):
        cand = RepoCandidate(
            repo_name="maziarraissi/PINNs",
            description="Physics-informed neural networks",
            repo_urls=["https://github.com/maziarraissi/PINNs"])
        # tokens: physics/informed/neural/networks -> >=1 命中
        assert is_trusted_repo_candidate(
            "physics informed neural networks pinn", cand)

    def test_no_overlap_rejected(self):
        cand = RepoCandidate(
            repo_name="torvalds/linux",
            description="Linux kernel source",
            repo_urls=["https://github.com/torvalds/linux"])
        assert not is_trusted_repo_candidate(
            "physics informed neural networks", cand)


class TestGithubScore:
    def test_name_and_desc_hits(self):
        score = github_repo_search_score(
            "attention is all you need",
            "harvardnlp/annotated-transformer",
            "Annotated PyTorch implementation of the Transformer",
            42000)
        assert score > 10

    def test_stars_bucket(self):
        base = github_repo_search_score("unet", "a/b", "none", 0)
        assert github_repo_search_score("unet", "a/b", "none", 99999) > base


class TestSignificantTokens:
    def test_stopwords_filtered(self):
        toks = significant_tokens("Attention is all you need paper")
        assert "attention" in toks
        assert "is" not in toks and "paper" not in toks


class TestReproMode:
    def test_auto_degrades_to_smoke(self):
        m = decide_reproduction_mode(requested="auto")
        assert m["effective_mode"] == "smoke"
        assert "auto" in m["reason"]

    def test_full_requires_confirmation(self):
        m = decide_reproduction_mode(requested="full",
                                     full_requested=False)
        assert m["effective_mode"] == "smoke"

    def test_full_confirmed(self):
        m = decide_reproduction_mode(requested="full",
                                     full_requested=True)
        assert m["effective_mode"] == "full"

    def test_explicit_smoke(self):
        m = decide_reproduction_mode(requested="smoke")
        assert m["effective_mode"] == "smoke"
        assert m["reason"] == "explicit"


# ---------------- discover_repositories 四层降级 ----------------

class _FakeProc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestDiscoveryChain:
    def test_offline_skips_network_layers(self, monkeypatch):
        """offline=True 时不发起 PwC/GitHub 请求，仅 curated 可用。"""
        hit = []

        def fake_http(url, timeout, headers=None):
            hit.append(url)
            raise AssertionError("offline 模式不应发起网络请求")

        monkeypatch.setattr(rd, "_http_json", fake_http)
        result = discover_repositories("pinn tutorials", offline=True)
        assert "github_search" in result["discovery_chain"]
        assert "curated_fallback" in result["discovery_chain"]
        assert result["selected_repo"].endswith("/maziarraissi/PINNs")
        assert hit == []
        offline = discover_repositories(
            "quantum entropy reproduction", offline=True)
        assert offline["selected_repo"] == ""

    def test_user_preference_short_circuit(self, monkeypatch):
        """用户 URL 直达，不触网络。"""
        hit = []

        def fake_http(url, timeout, headers=None):
            hit.append(url)
            raise AssertionError("user_preference 不应发起网络请求")

        monkeypatch.setattr(rd, "_http_json", fake_http)
        result = discover_repositories(
            "anything", preferred_url="https://github.com/owner/repo")
        assert result["discovery_chain"] == ["user_preference"]
        assert result["selected_repo"] == "https://github.com/owner/repo"
        assert hit == []

    def test_pwc_hit(self, monkeypatch):
        """PwC 返回带仓库的论文，跳过 GitHub/curated。"""
        def fake_http(url, timeout, headers=None):
            if url.endswith("/api/papers/search?q=pinn"):
                return [{"paper": {"id": "pinn-paper",
                                   "title": "Physics-informed neural networks"}}], ""
            if "/api/papers/pinn-paper" in url:
                return {"code": "https://github.com/maziarraissi/PINNs"}, ""
            return None, "unexpected url " + url

        monkeypatch.setattr(rd, "_http_json", fake_http)
        result = discover_repositories("pinn")
        assert "papers_with_code" in result["discovery_chain"]
        assert result["selected_repo"] == "https://github.com/maziarraissi/PINNs"
        assert result["fallback_used"] is False

    def test_github_fallback(self, monkeypatch):
        """PwC 失败 -> GitHub 搜索回退。"""
        def fake_http(url, timeout, headers=None):
            if "/api/papers/search" in url:
                return None, "boom pwc"
            if "api.github.com/search/repositories" in url:
                return {"items": [{
                    "full_name": "maziarraissi/PINNs",
                    "name": "PINNs",
                    "description": "Physics-informed neural networks",
                    "stargazers_count": 4550,
                }]}, ""
            return None, "unexpected url " + url

        monkeypatch.setattr(rd, "_http_json", fake_http)
        result = discover_repositories(
            "physics informed neural networks implementation")
        assert "github_search" in result["discovery_chain"]
        assert result["selected_repo"].endswith("/maziarraissi/PINNs")
        assert result["fallback_used"] is True

    def test_curated_ultimate_fallback(self, monkeypatch):
        """PwC + GitHub 全失败 -> curated 兜底。"""
        def fake_http(url, timeout, headers=None):
            if "github" in url or "papers" in url:
                return None, "all down"
            return None, "unexpected url " + url

        monkeypatch.setattr(rd, "_http_json", fake_http)
        result = discover_repositories("pinn tutorials")
        assert "curated_fallback" in result["discovery_chain"]
        assert result["selected_repo"].endswith("/maziarraissi/PINNs")
        assert result["fallback_used"] is True

    def test_total_failure_is_graceful(self, monkeypatch):
        def fake_http(url, timeout, headers=None):
            return None, "network down"

        monkeypatch.setattr(rd, "_http_json", fake_http)
        result = discover_repositories("quantum entropy reproduction")
        assert result["selected_repo"] == ""
        assert "network down" in result["error"]
        assert result["discovery_chain"]


# ---------------- ResourceFinderAgent 融合 ----------------

class _JsonLLM:
    """返回可解析 JSON 的假 LLM，记录调用次数。"""

    def __init__(self, payload):
        self.payload = payload
        self.call_count = 0

    def chat(self, prompt, system_prompt="", temperature=0.3, task=""):
        self.call_count += 1
        return json.dumps(self.payload, ensure_ascii=False)

    def get_call_count(self):
        return self.call_count


class TestResourceFinderFusion:
    def test_preferred_url_wins_and_chain_recorded(self):
        llm = _JsonLLM({"code_repo_url": "https://github.com/llm/guess",
                        "alternative_repos": [],
                        "dataset_url": "CIFAR-10",
                        "weights_url": "https://huggingface.co/x/y",
                        "confidence": 0.5})
        agent = ResourceFinderAgent(llm)
        result = agent.run({
            "paper_info": {"title": "Physics informed neural nets"},
            "preferred_repo_url": "https://github.com/user/pinn",
        })
        res = result["resources"]
        assert res["code_repo_url"] == "https://github.com/user/pinn"
        assert res["selected_repo"] == "https://github.com/user/pinn"
        assert res["repo_discovery"]["discovery_chain"] == ["user_preference"]
        assert res["repro_mode"]["effective_mode"] == "smoke"
        assert res["confidence"] >= 0.8

    def test_llm_fallback_with_no_selection(self, monkeypatch):
        """无用户 URL 且网络全失败时，LLM 猜测作为兜底保留。"""
        monkeypatch.setattr(
            rd, "_http_json",
            lambda url, timeout, headers=None: (None, "offline"))
        llm = _JsonLLM({"code_repo_url": "https://github.com/llm/guess",
                        "dataset_url": "MNIST",
                        "confidence": 0.4})
        agent = ResourceFinderAgent(llm)
        result = agent.run({
            "paper_info": {"title": "quantum entropy reproduction"},
            "raw_text": "some text without urls",
        })
        res = result["resources"]
        assert res["code_repo_url"] == "https://github.com/llm/guess"
        assert res["selected_repo"] == "https://github.com/llm/guess"

    def test_raw_text_github_url_bridges(self, monkeypatch):
        """确定性链未命中时，文本中的真实 GitHub URL 作为补充采纳。"""
        monkeypatch.setattr(
            rd, "_http_json",
            lambda url, timeout, headers=None: (None, "offline"))
        llm = _JsonLLM({"code_repo_url": "",
                        "dataset_url": "",
                        "confidence": 0.0})
        agent = ResourceFinderAgent(llm)
        raw = ("code released at https://github.com/owner/repo "
               "under MIT license")
        result = agent.run({"paper_info": {"title": "unknown title"},
                            "raw_text": raw})
        res = result["resources"]
        assert res["code_repo_url"] == "https://github.com/owner/repo"
        assert "https://github.com/owner/repo" in res["extracted_urls"]

    def test_pinned_revision_from_paper_info_and_text(self):
        llm = _JsonLLM({"code_repo_url": "", "confidence": 0.0})
        agent = ResourceFinderAgent(llm)
        result = agent.run({
            "paper_info": {"title": "x",
                           "code_revision": "abcd1234"},
            "raw_text": "see https://github.com/a/b/commit/ef0123456789abc more",
        })
        discovery = result["resources"]["repo_discovery"]
        assert discovery["pinned_revision"] == "abcd1234"

    def test_extract_pinned_revision_rules(self):
        agent = ResourceFinderAgent(_JsonLLM({}))
        assert agent._extract_pinned_revision({}, "") == ""
        assert agent._extract_pinned_revision(
            {"revision": "not-a-sha"}, "") == ""
        got = agent._extract_pinned_revision(
            {}, "commit f00df00df00df00df00df00df00df00df00df00d done")
        assert got == "f00df00df00df00df00df00df00df00df00df00d"


# ---------------- fetch_code revision pin + 溯源标记 ----------------

class TestFetchCodePin:
    def _fake_run_recorder(self):
        calls = []

        def fake_run(cmd, capture_output=None, text=None, timeout=None,
                     **kwargs):
            calls.append(list(cmd))
            if "clone" in cmd:
                return _FakeProc(0)
            if "checkout" in cmd:
                return _FakeProc(0)
            if "fetch" in cmd:
                return _FakeProc(0, "", "fetched")
            if "rev-parse" in cmd:
                # 第一次 HEAD=deadbeef（pin 前），第二次 cafebabe（pin 后）
                head = "cafebabe" if len(calls) > 2 else "deadbeef"
                return _FakeProc(0, head + "\n")
            return _FakeProc(0)
        return calls, fake_run

    def test_clone_pins_revision_and_writes_marker(self, tmp_path,
                                                   monkeypatch):
        calls, fake_run = self._fake_run_recorder()
        monkeypatch.setattr(subprocess, "run", fake_run)
        rm = ResourceManager(data_root=str(tmp_path / "data"))
        info = rm.fetch_code(
            "p1", "https://github.com/owner/repo",
            target=str(tmp_path / "repo"), revision="abcd1234")

        assert info["state"] == "cloned"
        assert info["commit"] == "cafebabe"       # pin 后 HEAD
        assert info["revision"] == "abcd1234"
        assert "pinned revision abcd1234" in info["detail"]

        cmds = [c for c in calls if "clone" in c]
        assert cmds[0][:5] == ["git", "clone", "--depth", "1",
                               "--single-branch"]

        marker = tmp_path / "repo" / ".autorepro-repo-source.json"
        assert marker.exists()
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["repo_url"] == "https://github.com/owner/repo"
        assert payload["commit"] == "cafebabe"
        assert payload["revision"] == "abcd1234"
        assert payload["acquisition"] == "cloned"

    def test_pin_failure_degrades_to_head(self, tmp_path, monkeypatch):
        calls, fake_run = self._fake_run_recorder()
        real_fetch = fake_run

        def fake_run_fail(cmd, **kwargs):
            if "fetch" in cmd:
                return _FakeProc(1, "", "unknown revision abcd1234")
            return real_fetch(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", fake_run_fail)
        rm = ResourceManager(data_root=str(tmp_path / "data"))
        info = rm.fetch_code(
            "p2", "https://github.com/owner/repo",
            target=str(tmp_path / "repo2"), revision="abcd1234")
        assert info["state"] == "cloned"
        assert info["commit"] == "deadbeef"       # 保留 HEAD
        assert "pin 失败" in info["detail"]

    def test_cached_reuse_writes_marker(self, tmp_path, monkeypatch):
        _, fake_run = self._fake_run_recorder()
        monkeypatch.setattr(subprocess, "run", fake_run)
        repo = tmp_path / "repo3"
        repo.mkdir(parents=True)
        (repo / "run.py").write_text("print(1)", encoding="utf-8")
        rm = ResourceManager(data_root=str(tmp_path / "data"))
        info = rm.fetch_code("p3", "https://github.com/owner/repo",
                             target=str(repo), revision="beef1234")
        assert info["state"] == "cached"
        marker = repo / ".autorepro-repo-source.json"
        assert marker.exists()
        assert json.loads(marker.read_text(encoding="utf-8"))[
            "acquisition"] == "cached"

    def test_no_revision_skips_pin(self, tmp_path, monkeypatch):
        calls, fake_run = self._fake_run_recorder()
        monkeypatch.setattr(subprocess, "run", fake_run)
        rm = ResourceManager(data_root=str(tmp_path / "data"))
        info = rm.fetch_code("p4", "https://github.com/owner/repo",
                             target=str(tmp_path / "repo4"))
        assert info["state"] == "cloned"
        assert info["commit"] == "deadbeef"
        fetch_calls = [c for c in calls if "fetch" in c]
        assert fetch_calls == []   # 未请求 pin 时不发 fetch
"""P1-⑨ 确定性仓库发现链测试。

覆盖：纯函数（URL 归一/提取/查询构造/curated/信任判定/评分/模式决策）、
discover_repositories 四层降级链（monkeypatch 网络层）、
ResourceFinderAgent 融合（确定性优先 + pin revision + repro_mode）、
resource_manager.fetch_code revision pin + 溯源标记写入。
全部避免真实网络。
"""
import json
import subprocess
import urllib.parse

import pytest

from src.agents import repo_discovery as rd
from src.agents.repo_discovery import (
    RepoCandidate,
    build_repo_discovery_query,
    build_github_fallback_queries,
    build_github_readme_query,
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


@pytest.fixture(autouse=True)
def isolate_web_fallback(monkeypatch):
    # These legacy chain tests inject JSON networking only. The public-page
    # fallback has separate network-isolated parser and integration coverage.
    monkeypatch.setattr(rd, '_search_github_web', lambda *a, **k: ([], 'web search unavailable'))


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


class TestExactTitleReadmeDiscovery:
    title = 'Continuous Vector Operators'

    @staticmethod
    def row(name='authors/CVO', description=None, stars=1):
        return {'full_name': name, 'description': description, 'stargazers_count': stars}

    def test_complete_title_is_quoted_and_internal_query_syntax_is_literal(self):
        assert build_github_readme_query('  "Continuous\n Vector Operators"  ') == '"Continuous Vector Operators" in:readme'
        assert build_github_readme_query('Alpha "Beta" \\ Gamma') == '"Alpha \\"Beta\\" \\\\ Gamma" in:readme'
        assert len(build_github_readme_query('long '+('a '*500))) <= 214
        for value in ('', 'PINN', '2106.09685', '2106.09685v2'):
            assert build_github_readme_query(value) == ''

    def test_observed_readme_rank_wins_without_description_or_metadata_title_tokens(self, monkeypatch):
        requests = []

        def http(url, timeout, headers=None):
            requests.append(url)
            return {'items': [self.row(), self.row('other/Survey', self.title, 999999)]}, ''

        monkeypatch.setattr(rd, '_http_json', http)
        candidates, error = rd._search_github(self.title)
        assert not error and len(requests) == 1
        parameters = urllib.parse.parse_qs(urllib.parse.urlsplit(requests[0]).query)
        assert parameters == {'q': ['"Continuous Vector Operators" in:readme'], 'per_page': ['5']}
        assert [row.repo_name for row in candidates] == ['authors/CVO', 'other/Survey']
        assert [row.search_rank for row in candidates] == [1, 2]
        assert candidates[0].score_hint > candidates[1].score_hint
        assert candidates[0].description == '' and candidates[0].stars == 1
        assert candidates[0].source == 'github_search_readme'
        assert candidates[0].evidence_url == requests[0]
        assert is_trusted_repo_candidate(self.title, candidates[0])
        assert not is_trusted_repo_candidate('Different Paper Title', candidates[0])
        assert candidates[0].to_dict()['search_query'] == parameters['q'][0]

    @pytest.mark.parametrize('exact_response,exact_error', [
        ({'items': []}, ''), (None, 'timeout'), ({'items': 'malformed'}, ''),
        ({'items': [None, {'full_name': '../escape'}, {'full_name': 'a/private', 'private': True}]}, ''),
    ])
    def test_empty_failed_or_invalid_exact_results_retain_original_broad_fallback(self, monkeypatch, exact_response, exact_error):
        requests = []

        def http(url, timeout, headers=None):
            requests.append(url)
            if 'sort=stars' not in url:
                return exact_response, exact_error
            return {'items': [self.row('other/Implementation', self.title)]}, ''

        monkeypatch.setattr(rd, '_http_json', http)
        candidates, error = rd._search_github(self.title)
        assert len(requests) == 2 and 'sort=stars&order=desc' in requests[1]
        assert candidates[0].source == 'github_search'
        assert candidates[0].repo_name == 'other/Implementation'
        assert ('timeout' in error) == bool(exact_error)

    def test_deduplicates_and_preserves_original_search_ranks(self, monkeypatch):
        monkeypatch.setattr(rd, '_http_json', lambda *args, **kwargs: ({'items': [
            self.row(), self.row('AUTHORS/cvo'), self.row('other/Repo')]}, ''))
        candidates, _ = rd._search_github(self.title)
        assert [row.search_rank for row in candidates] == [1, 3]

    @pytest.mark.parametrize('items', ['malformed', [None, 'not a repository', {'full_name': '../escape'}], []])
    def test_malformed_metadata_after_exact_failure_retains_web_fallback(self, monkeypatch, items):
        def http(url, timeout, headers=None):
            return ({'items': items}, '') if 'sort=stars' in url else (None, 'exact timeout')

        monkeypatch.setattr(rd, '_http_json', http)
        candidates, error = rd._search_github(self.title)
        assert candidates == []
        assert 'exact timeout' in error and 'web search unavailable' in error

    def test_metadata_fallback_does_not_coerce_malformed_description_or_stars(self, monkeypatch):
        def http(url, timeout, headers=None):
            return ({'items': [self.row(description={'invalid': True}, stars='invalid')]}, '') if 'sort=stars' in url else ({'items': []}, '')

        monkeypatch.setattr(rd, '_http_json', http)
        candidates, _ = rd._search_github(self.title)
        assert candidates[0].description == '' and candidates[0].stars == 0

    def test_arxiv_and_single_word_queries_keep_old_network_path(self, monkeypatch):
        requests = []

        def http(url, timeout, headers=None):
            requests.append(url)
            return {'items': [self.row('authors/Implementation', 'test')]}, ''

        monkeypatch.setattr(rd, '_http_json', http)
        for query in ['2106.09685v2', 'PINN']:
            candidates, _ = rd._search_github(query)
            assert candidates[0].source == 'github_search'
        assert len(requests) == 2 and all('sort=stars' in url for url in requests)

    @pytest.mark.parametrize('change', [
        {'evidence_url': ''}, {'evidence_url': 'https://api.github.com/search/repositories?q=other'},
        {'evidence_url': 'https://github.com/search?q=other'},
        {'evidence_url': 'https://[invalid'}, {'search_query': 'Different Paper'},
        {'search_rank': 0}, {'search_rank': True}, {'repo_urls': []},
    ])
    def test_unobserved_source_marker_cannot_bypass_candidate_filter(self, change):
        query = build_github_readme_query(self.title)
        values = {'source': 'github_search_readme', 'repo_urls': ['https://github.com/a/b'],
                  'search_query': query, 'search_rank': 1,
                  'evidence_url': 'https://api.github.com/search/repositories?'+urllib.parse.urlencode({'q': query})}
        assert not is_trusted_repo_candidate(self.title, RepoCandidate(**{**values, **change}))

    def test_discovery_and_resource_identity_preserve_search_candidate_status(self, monkeypatch):
        def http(url, timeout, headers=None):
            if 'api.github.com' in url:
                return {'items': [self.row()]}, ''
            return [], ''

        monkeypatch.setattr(rd, '_http_json', http)
        discovery = discover_repositories(self.title)
        assert discovery['selected_repo'] == 'https://github.com/authors/CVO'
        assert discovery['discovery_chain'][-1] == 'github_search_readme'
        assert discovery['candidates'][0]['search_rank'] == 1
        agent = ResourceFinderAgent(_JsonLLM({'code_repo_url': '', 'confidence': 1.0}))
        resources = agent.run({'paper_info': {'title': self.title}})['resources']
        assert resources['code_repo_url'] == discovery['selected_repo']
        assert resources['repository_identity']['status'] == 'candidate_unverified'
        assert resources['repository_identity']['is_official'] is None
        assert resources['repository_identity']['repository_executed'] is False
        assert resources['repository_identity']['source'] == 'github_search_readme'
        assert resources['confidence'] <= 0.5

    def test_explicit_pdf_author_statement_bypasses_search(self, monkeypatch):
        def unexpected(*args, **kwargs):
            raise AssertionError('An explicit PDF author statement must retain priority')

        monkeypatch.setattr(rd, '_http_json', unexpected)
        resources = ResourceFinderAgent(_JsonLLM({'code_repo_url': 'https://github.com/other/Guess'})).run({
            'paper_info': {'title': self.title}, 'extracted_repository_links': [{
                'url': 'https://github.com/authors/Original', 'page': 1,
                'source': 'pdf_text', 'is_author_code': True, 'evidence_type': 'author_code_statement'}]})['resources']
        assert resources['code_repo_url'] == 'https://github.com/authors/Original'

    @pytest.mark.parametrize('pwc_id', [None, '2106.09686', '2106.09685'])
    def test_arxiv_id_is_for_paper_lookup_and_title_is_for_repository_search(self, monkeypatch, pwc_id):
        paper_id = '2106.09685v2'
        pwc_repo = 'https://github.com/pwc/Result'
        requested = []

        def http(url, timeout, headers=None):
            parsed_url = urllib.parse.urlsplit(url)
            query = urllib.parse.parse_qs(parsed_url.query).get('q', [''])[0]
            if parsed_url.path.endswith('/papers/search'):
                requested.append(query)
                return ([{'paper': {'id': pwc_id}}] if pwc_id else []), ''
            if parsed_url.path.endswith('/search/repositories'):
                requested.append(query)
                return {'items': [self.row()] if query == build_github_readme_query(self.title)
                        else [self.row('other/Citation')]}, ''
            return {'code': pwc_repo}, ''

        monkeypatch.setattr(rd, '_http_json', http)
        resources = ResourceFinderAgent(_JsonLLM({})).run({'paper_info': {
            'title': self.title, 'arxiv_id': paper_id}})['resources']
        discovery = resources['repo_discovery']
        assert discovery['query'] == paper_id
        assert resources['repository_identity']['status'] == 'candidate_unverified'
        assert resources['repository_identity']['is_official'] is None
        if pwc_id == '2106.09685':
            assert requested == [paper_id]
            assert resources['selected_repo'] == pwc_repo
        else:
            assert requested == [paper_id, build_github_readme_query(self.title)]
            assert resources['selected_repo'] == 'https://github.com/authors/CVO'
            assert discovery['candidates'][0]['search_rank'] == 1
            assert pwc_repo not in resources['alternative_repos']

    def test_arxiv_id_does_not_bypass_relevance_filter_on_title_results(self, monkeypatch):
        monkeypatch.setattr(rd, '_search_pwc', lambda *args, **kwargs: ([], ''))
        monkeypatch.setattr(rd, '_search_github', lambda *args, **kwargs: ([RepoCandidate(
            source='github_search', repo_urls=['https://github.com/other/Citation'])], ''))
        result = discover_repositories('2106.09685', paper_title=self.title)
        assert not result['selected_repo']


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
    def _fake_snapshot(self, tmp_path, monkeypatch):
        from src import repository_reproduction as reproduction

        resolved = "a" * 40
        requests = []

        def resolve(root, url, revision=""):
            requests.append((url, revision))
            return resolved

        def export(root, profile, destination, **kwargs):
            destination.mkdir(parents=True)
            (destination / "run.py").write_text("print(1)", encoding="utf-8")
            assert profile["repository"]["revision"] == resolved
            return {"url": profile["repository"]["url"], "path": str(destination),
                    "resolved_sha": resolved, "acquisition": "cloned",
                    "cache_path": str(tmp_path / "source-cache"), "archive_sha256": "b" * 64,
                    "files": {"run.py": "c" * 64}}

        monkeypatch.setattr(reproduction, "resolve_repository_revision", resolve)
        monkeypatch.setattr(reproduction, "export_repository", export)
        return requests, resolved

    def test_clone_pins_revision_and_writes_marker(self, tmp_path, monkeypatch):
        requests, resolved = self._fake_snapshot(tmp_path, monkeypatch)
        manager = ResourceManager(data_root=str(tmp_path / "data"))
        info = manager.fetch_code("p1", "https://github.com/owner/repo",
                                  target=str(tmp_path / "repo"), revision="abcd1234")
        assert requests == [("https://github.com/owner/repo", "abcd1234")]
        assert info["state"] == "cloned"
        assert info["commit"] == resolved
        payload = json.loads((tmp_path / "repo" / ".autorepro-repo-source.json").read_text(encoding="utf-8"))
        assert payload["repo_url"] == "https://github.com/owner/repo"
        assert payload["revision"] == "abcd1234"
        assert payload["commit"] == resolved
        assert payload["archive_sha256"] == "b" * 64

    def test_pin_failure_returns_no_usable_source(self, tmp_path, monkeypatch):
        from src import repository_reproduction as reproduction
        self._fake_snapshot(tmp_path, monkeypatch)

        def fail(*args, **kwargs):
            raise RuntimeError("unknown requested revision")

        monkeypatch.setattr(reproduction, "export_repository", fail)
        manager = ResourceManager(data_root=str(tmp_path / "data"))
        info = manager.fetch_code("p2", "https://github.com/owner/repo", revision="abcd1234")
        assert info["state"] == "pin-failed"
        assert info["path"] == info["commit"] == ""
        assert "unknown requested revision" in info["detail"]

    def test_existing_explicit_target_is_preserved(self, tmp_path, monkeypatch):
        requests, _ = self._fake_snapshot(tmp_path, monkeypatch)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "run.py").write_text("original", encoding="utf-8")
        manager = ResourceManager(data_root=str(tmp_path / "data"))
        info = manager.fetch_code("p3", "https://github.com/owner/repo",
                                  target=str(repo), revision="beef1234")
        assert info["state"] == "pin-failed"
        assert info["path"] == ""
        assert not requests
        assert (repo / "run.py").read_text(encoding="utf-8") == "original"
        assert not (repo / ".autorepro-repo-source.json").exists()

    def test_no_revision_is_resolved_before_export(self, tmp_path, monkeypatch):
        requests, resolved = self._fake_snapshot(tmp_path, monkeypatch)
        manager = ResourceManager(data_root=str(tmp_path / "data"))
        info = manager.fetch_code("p4", "https://github.com/owner/repo")
        assert requests == [("https://github.com/owner/repo", "")]
        assert info["state"] == "cloned"
        assert info["commit"] == resolved
        assert info["revision"] == ""

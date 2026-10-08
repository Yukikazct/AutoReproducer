"""Public source provenance/containment tests; no HTTP, API, or training."""
import hashlib
import json
from unittest.mock import Mock, call

import pytest

from src.repository_profiles import get_profile
from src.repository_public_sources import (
    AUTHOR_COMMENTS,
    MAX_COMMENT_BYTES,
    MAX_DOCUMENT_BYTES,
    PAPER_SECTIONS,
    PAPER_URL,
    PUBLIC_REPOSITORY_FILES,
    RepositoryPublicSources,
    download_public_document,
    extract_author_comment,
    extract_paper_sections,
)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def comment_document(comment_id, *, body=None):
    spec = AUTHOR_COMMENTS[comment_id]
    return json.dumps({"id": comment_id, "url": spec["api_url"],
                       "html_url": spec["html_url"], "issue_url": spec["issue_url"],
                       "user": {"login": "public-author", "private_metadata": "UNSENT_USER_SECRET"},
                       "author_association": "MEMBER",
                       "body": body or f"New fetched author statement {comment_id}.\r\nOriginal public text.",
                       "unrelated_metadata": "UNSENT_ENVELOPE_SECRET"}).encode()


def public_transport(paper_html):
    documents = {PAPER_URL: paper_html, **{spec["api_url"]: comment_document(comment_id)
                                         for comment_id, spec in AUTHOR_COMMENTS.items()}}
    return Mock(side_effect=lambda url, **kwargs: documents[url])


def seed_comments(directory):
    paths = {}
    for comment_id in AUTHOR_COMMENTS:
        path = directory / f"public_comment_{comment_id}.json"
        path.write_bytes(comment_document(comment_id))
        paths[comment_id] = path
    return paths


@pytest.fixture
def paper_html():
    # Unexpected fixture values prove that excerpts come from HTML, rather than
    # an embedded summary or the application's reference metric constants.
    return b'''<!doctype html><html><head><script>HEAD_SECRET</script></head><body>
<div>arXiv:2205.13504v3 [cs.AI] 17 Aug 2022</div>
<article><h1>Are Transformers Effective for Time Series Forecasting?</h1>
<div id="abstract1"><h6>Abstract</h6><p>Fresh public abstract &amp; evidence.</p></div>
<section id="S4"><h2>An Embarrassingly Simple Baseline</h2>
<p>Newly fetched decomposition description. Kernel = 25.</p>
<script>SCRIPT_SECRET</script></section>
<section id="S5.SS1"><h3>Experimental Settings</h3><p>Use MSE and MAE.</p></section>
<section id="S5.SS2"><h3>Comparison</h3><p>Read Table 2 directly.</p></section>
<figure id="S5.T2"><table>
<tr><td colspan="2">Methods</td><td colspan="2">DLinear*</td></tr>
<tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td></tr>
<tr><td rowspan="2">ETTh1</td><td>96</td><td>0.123</td><td>0.234</td></tr>
<tr><td>192</td><td>0.345</td><td>0.456</td></tr>
</table><figcaption>Table 2: Multivariate errors, lower is better.</figcaption></figure>
<section id="A2.SS2"><h3>B.2 Implementation Details</h3><p>Report
<math alttext="L=336"><mi>L</mi><mo>=</mo><mn>336</mn>
<annotation encoding="application/x-tex">DUPLICATE_MATH_SECRET</annotation></math>.
New public implementation evidence.</p></section></article></body></html>'''


@pytest.fixture
def public_workspace(tmp_path):
    profile = get_profile("dlinear_etth1_reference")
    root = tmp_path / "exported"
    root.mkdir()
    hashes = {}
    for relative in PUBLIC_REPOSITORY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"# real exported public fixture: {relative}\npublic_fact = 2021\n".encode()
        path.write_bytes(content)
        hashes[relative] = digest(content)
    # A malicious extra snapshot entry must not become an API source.
    for relative in ("results/private_metrics.json", "logs/run.log", ".env",
                     "utils/private_runtime.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PRIVATE_SECRET_AND_LOCAL_RESULTS", encoding="utf-8")
        hashes[relative] = digest(path.read_bytes())
    snapshot = {**profile["repository"], "resolved_sha": profile["repository"]["revision"],
                "path": str(root), "cache_path": "/local/private/cache", "files": hashes}
    profile["credentials"] = "PRIVATE_PROFILE_SECRET"
    profile["local_metrics"] = {"mse": -12345}
    return root, snapshot, profile


def build(tmp_path, paper_html, public_workspace, **kwargs):
    root, snapshot, profile = public_workspace
    transport = public_transport(paper_html)
    builder = RepositoryPublicSources(tmp_path / "public_cache", transport=transport)
    packet = builder.build_packet(root, snapshot, profile, offline=False, **kwargs)
    return builder, packet, transport


def test_packet_contains_actual_html_and_only_the_approved_export(public_workspace, paper_html, tmp_path):
    builder, packet, transport = build(tmp_path, paper_html, public_workspace)
    assert set(packet) == {"version", "repository", "sources"}
    assert packet["repository"] == get_profile("dlinear_etth1_reference")["repository"]
    assert len(packet["sources"]) == len(PAPER_SECTIONS) + len(PUBLIC_REPOSITORY_FILES) + len(AUTHOR_COMMENTS)
    assert all(set(source) == {"source_id", "url", "locator", "text"}
               for source in packet["sources"])
    rendered = json.dumps(packet)
    assert "Newly fetched decomposition description" in rendered
    assert "PRIVATE_SECRET" not in rendered
    assert "PRIVATE_PROFILE_SECRET" not in rendered
    assert "PRIVATE_SECRET_AND_LOCAL_RESULTS" not in rendered
    assert str(public_workspace[0]) not in rendered
    assert "/local/private/cache" not in rendered
    assert "local_metrics" not in rendered
    assert "SCRIPT_SECRET" not in rendered
    assert "HEAD_SECRET" not in rendered
    assert "DUPLICATE_MATH_SECRET" not in rendered
    assert "UNSENT_ENVELOPE_SECRET" not in rendered
    assert "UNSENT_USER_SECRET" not in rendered
    assert "L=336" in rendered
    assert ".375" not in rendered
    assert "ETTh1 | 96 | 0.123 | 0.234" in rendered
    assert "ETTh1 | 192 | 0.345 | 0.456" in rendered
    assert transport.call_args_list == [call(PAPER_URL, max_bytes=MAX_DOCUMENT_BYTES, timeout_s=30),
                                       *(call(spec["api_url"], max_bytes=MAX_COMMENT_BYTES, timeout_s=30)
                                         for spec in AUTHOR_COMMENTS.values())]
    for source in packet["sources"]:
        if source["source_id"].startswith("repo_"):
            assert packet["repository"]["revision"] in source["url"]
            assert source["text"] == (public_workspace[0] / source["locator"].split("#")[0]).read_text(encoding="utf-8")
    assert builder.manifest_path.is_file()


def test_manifest_records_document_section_and_source_hashes(public_workspace, paper_html, tmp_path):
    builder, packet, _ = build(tmp_path, paper_html, public_workspace)
    manifest = json.loads(builder.manifest_path.read_text(encoding="utf-8"))
    assert manifest["paper"]["url"] == PAPER_URL
    assert manifest["paper"]["sha256"] == digest(paper_html)
    assert manifest["paper"]["bytes"] == len(paper_html)
    sources = {source["source_id"]: source for source in packet["sources"]}
    for section in manifest["paper_sections"]:
        text = sources[section["source_id"]]["text"]
        assert section["text_sha256"] == digest(text.encode())
        assert section["html_lines"][0] <= section["html_lines"][1]
        assert section["characters"] == len(text)
    assert {item["locator"].split("#")[0] for item in manifest["repository_files"]} == set(PUBLIC_REPOSITORY_FILES)
    for item in manifest["repository_files"]:
        assert item["sha256"] == public_workspace[1]["files"][item["locator"].split("#")[0]]
    expected = digest(json.dumps(packet, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":")).encode())
    assert manifest["packet_sha256"] == expected
    assert str(public_workspace[0]) not in json.dumps(manifest)
    assert {item["source_id"] for item in manifest["author_comments"]} == {"author_single_seed", "author_initialization"}
    for item in manifest["author_comments"]:
        assert item["sha256"] == digest(comment_document(item["id"]))
        assert item["text_sha256"] == digest(sources[item["source_id"]]["text"].encode())


def test_verified_local_seed_then_offline_cache_never_uses_transport(public_workspace, paper_html, tmp_path):
    seed = tmp_path / "verified_public.html"
    seed.write_bytes(paper_html)
    forbidden_transport = Mock(side_effect=AssertionError("offline transport was called"))
    builder = RepositoryPublicSources(tmp_path / "cache", transport=forbidden_transport)
    root, snapshot, profile = public_workspace
    first = builder.build_packet(root, snapshot, profile, offline=True, paper_path=seed,
                                 comment_paths=seed_comments(tmp_path))
    seed.unlink()
    second = builder.build_packet(root, snapshot, profile, offline=True)
    assert second == first
    forbidden_transport.assert_not_called()
    assert (builder.cache_dir / "paper.html").read_bytes() == paper_html
    assert json.loads((builder.cache_dir / "paper_manifest.json").read_text(encoding="utf-8"))["origin"] == "verified_local_public_document"


def test_offline_missing_cache_fails_without_transport(public_workspace, tmp_path):
    transport = Mock()
    builder = RepositoryPublicSources(tmp_path / "missing", transport=transport)
    with pytest.raises(RuntimeError, match="离线"):
        builder.build_packet(*public_workspace, offline=True)
    transport.assert_not_called()
    assert not builder.manifest_path.exists()


@pytest.mark.parametrize("tamper", ["document", "url", "hash", "bytes", "missing_manifest"])
def test_cache_tampering_is_rejected_not_replaced_by_a_download(public_workspace, paper_html, tmp_path, tamper):
    builder, _, transport = build(tmp_path, paper_html, public_workspace)
    transport.reset_mock()
    path = builder.cache_dir / "paper_manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "document":
        (builder.cache_dir / "paper.html").write_bytes(paper_html + b"tampered")
    elif tamper == "missing_manifest":
        path.unlink()
    else:
        metadata[{"url": "url", "hash": "sha256", "bytes": "bytes"}[tamper]] = "tampered"
        path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="缓存"):
        builder.build_packet(*public_workspace, offline=False)
    transport.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("url", "https://github.com/impostor/repo"),
    ("revision", "a" * 40),
    ("resolved_sha", "b" * 40),
])
def test_repository_identity_is_checked_before_network(public_workspace, tmp_path, field, value):
    root, snapshot, profile = public_workspace
    snapshot[field] = value
    transport = Mock()
    builder = RepositoryPublicSources(tmp_path / "cache", transport=transport)
    with pytest.raises(ValueError, match="固定作者仓库SHA"):
        builder.build_packet(root, snapshot, profile, offline=False)
    transport.assert_not_called()


@pytest.mark.parametrize("tamper", ["changed", "missing_hash", "invalid_hash", "missing_file"])
def test_every_allowed_file_must_match_snapshot(public_workspace, tmp_path, tamper):
    root, snapshot, profile = public_workspace
    relative = "models/DLinear.py"
    if tamper == "changed":
        (root / relative).write_text("LOCAL_SECRET_MODIFICATION", encoding="utf-8")
    elif tamper == "missing_file":
        (root / relative).unlink()
    elif tamper == "missing_hash":
        snapshot["files"].pop(relative)
    else:
        snapshot["files"][relative] = "invalid"
    transport = Mock()
    builder = RepositoryPublicSources(tmp_path / "cache", transport=transport)
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.build_packet(root, snapshot, profile, offline=False)
    transport.assert_not_called()
    assert not builder.manifest_path.exists()


def test_nested_snapshot_hash_entries_are_supported(public_workspace, paper_html, tmp_path):
    snapshot = public_workspace[1]
    snapshot["files"] = {relative: {"sha256": value, "content": b"untrusted copy"}
                         for relative, value in snapshot["files"].items()}
    _, packet, _ = build(tmp_path, paper_html, public_workspace)
    assert "untrusted copy" not in json.dumps(packet)


@pytest.mark.parametrize("outside", [True, False])
def test_symlink_source_is_rejected_even_when_hash_matches(public_workspace, tmp_path, outside):
    root, snapshot, profile = public_workspace
    relative = "models/DLinear.py"
    target = tmp_path / "outside.py" if outside else root / "inside.py"
    content = (root / relative).read_bytes()
    target.write_bytes(content)
    (root / relative).unlink()
    (root / relative).symlink_to(target)
    builder = RepositoryPublicSources(tmp_path / "cache", transport=Mock())
    with pytest.raises(ValueError, match="越界|symlink"):
        builder.build_packet(root, snapshot, profile, offline=False)


def test_symlink_parent_directory_is_rejected(public_workspace, tmp_path):
    root, snapshot, profile = public_workspace
    original = root / "exp"
    moved = root / "linked_exp"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    builder = RepositoryPublicSources(tmp_path / "cache", transport=Mock())
    with pytest.raises(ValueError, match="symlink"):
        builder.build_packet(root, snapshot, profile, offline=False)


@pytest.mark.parametrize("url", [
    "http://arxiv.org/html/2205.13504v3", "https://arxiv.org/html/2205.13504v3?redirect=x",
    "https://arxiv.org.evil.example/html/2205.13504v3", "file:///etc/passwd",
    "https://user:secret@arxiv.org/html/2205.13504v3", "https://127.0.0.1/",
])
def test_download_url_allowlist_applies_to_injected_transport(url):
    transport = Mock()
    with pytest.raises(ValueError, match="白名单"):
        download_public_document(url, transport=transport)
    transport.assert_not_called()


def test_download_offline_guard_applies_to_injected_transport():
    transport = Mock()
    with pytest.raises(RuntimeError, match="离线"):
        download_public_document(PAPER_URL, offline=True, transport=transport)
    transport.assert_not_called()


def test_download_hard_size_limit_and_bytes_contract():
    with pytest.raises(ValueError, match="字节上限"):
        download_public_document(PAPER_URL, max_bytes=8, transport=Mock(return_value=b"x" * 9))
    with pytest.raises(TypeError, match="bytes"):
        download_public_document(PAPER_URL, transport=Mock(return_value="not bytes"))
    assert download_public_document(PAPER_URL, max_bytes=8, transport=Mock(return_value=b"x" * 8)) == b"x" * 8


@pytest.mark.parametrize("wrong", ["missing_title", "missing_version", "wrong_version", "missing_anchor", "duplicate_anchor", "invalid_utf8"])
def test_invalid_paper_identity_or_sections_fail_before_cache_publication(public_workspace, paper_html, tmp_path, wrong):
    if wrong == "missing_title":
        paper_html = paper_html.replace(b"Are Transformers Effective for Time Series Forecasting?", b"Different paper")
    elif wrong == "missing_version":
        paper_html = paper_html.replace(b"arXiv:2205.13504v3", b"version is missing")
    elif wrong == "wrong_version":
        paper_html = paper_html.replace(b"arXiv:2205.13504v3", b"arXiv:2205.13504v2")
    elif wrong == "missing_anchor":
        paper_html = paper_html.replace(b'id="S5.T2"', b'id="Other.Table"')
    elif wrong == "duplicate_anchor":
        paper_html = paper_html.replace(b"</article>", b'<div id="S5.T2">duplicate</div></article>')
    else:
        paper_html += b"\xff"
    builder = RepositoryPublicSources(tmp_path / "cache", transport=Mock(return_value=paper_html))
    with pytest.raises(ValueError):
        builder.build_packet(*public_workspace, offline=False)
    assert not (builder.cache_dir / "paper.html").exists()
    assert not builder.manifest_path.exists()


def test_repository_file_and_packet_size_limits_fail_without_truncation(public_workspace, paper_html, tmp_path):
    builder = RepositoryPublicSources(tmp_path / "small_files", max_repository_file_bytes=8)
    with pytest.raises(ValueError, match="字节上限"):
        builder.build_packet(*public_workspace)
    builder = RepositoryPublicSources(tmp_path / "small_packet", max_packet_characters=8,
                                      transport=public_transport(paper_html))
    with pytest.raises(ValueError, match="不会截断证据"):
        builder.build_packet(*public_workspace, offline=False)
    assert not builder.manifest_path.exists()


def test_seed_size_limit_uses_the_same_document_guard(public_workspace, tmp_path):
    seed = tmp_path / "large.html"
    seed.write_bytes(b"x" * 9)
    builder = RepositoryPublicSources(tmp_path / "cache", max_document_bytes=8)
    with pytest.raises(ValueError, match="字节上限"):
        builder.build_packet(*public_workspace, paper_path=seed)
    assert not (builder.cache_dir / "paper.html").exists()


def test_parser_preserves_table_headers_and_caption(paper_html):
    sources, provenance = extract_paper_sections(paper_html)
    table = next(source for source in sources if source["source_id"] == "paper_table2")
    assert "Methods | Methods | DLinear* | DLinear*" in table["text"]
    assert "Metric | Metric | MSE | MAE" in table["text"]
    assert "Table 2: Multivariate errors, lower is better." in table["text"]
    assert len(provenance) == 6


def test_author_comments_are_complete_original_bodies_with_only_public_packet_fields(tmp_path):
    bodies = {1331937601: "Public single-seed body with fresh wording.\r\nLine 2.",
              1398345611: "Public initialization body.\nhttps://example.org/not-fetched.png"}
    documents = {spec["api_url"]: comment_document(comment_id, body=bodies[comment_id])
                 for comment_id, spec in AUTHOR_COMMENTS.items()}
    transport = Mock(side_effect=lambda url, **kwargs: documents[url])
    builder = RepositoryPublicSources(tmp_path / "cache", transport=transport)
    sources, manifests = builder.cache_author_comments(offline=False)
    assert len(sources) == 2
    for source, manifest in zip(sources, manifests):
        assert set(source) == {"source_id", "url", "locator", "text"}
        assert source["text"] == bodies[manifest["id"]]
        assert source["locator"] == f"#issuecomment-{manifest['id']}"
        assert source["url"] == AUTHOR_COMMENTS[manifest["id"]]["html_url"]
        assert manifest["sha256"] == digest(documents[manifest["url"]])
        assert json.loads((builder.cache_dir / f"issuecomment-{manifest['id']}.manifest.json").read_text(encoding="utf-8")) == manifest
    assert transport.call_count == 2  # Linked images/URLs never trigger requests.


def test_offline_author_comment_seeds_and_verified_reuse_without_transport(tmp_path):
    transport = Mock(side_effect=AssertionError("offline network used"))
    builder = RepositoryPublicSources(tmp_path / "cache", transport=transport)
    first = builder.cache_author_comments(offline=True, comment_paths=seed_comments(tmp_path))
    second = builder.cache_author_comments(offline=True)
    assert second == first
    transport.assert_not_called()
    assert all(manifest["origin"] == "verified_local_public_comment" for manifest in first[1])


def test_offline_missing_author_comment_never_uses_profile_paraphrase_or_transport(tmp_path):
    transport = Mock()
    builder = RepositoryPublicSources(tmp_path / "cache", transport=transport)
    with pytest.raises(RuntimeError, match="离线"):
        builder.cache_author_comments(offline=True)
    transport.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("id", 1331937602), ("id", True),
    ("url", "https://api.github.com/repos/impostor/repo/issues/comments/1331937601"),
    ("html_url", "https://github.com/cure-lab/LTSF-Linear/issues/39#issuecomment-1331937601"),
    ("issue_url", "https://api.github.com/repos/cure-lab/LTSF-Linear/issues/99"),
    ("author_association", "NONE"), ("user", {"login": ""}),
    ("body", ""), ("body", None),
])
def test_comment_identity_attribution_and_nonempty_body_are_required(field, value):
    comment = json.loads(comment_document(1331937601))
    comment[field] = value
    with pytest.raises(ValueError):
        extract_author_comment(json.dumps(comment).encode(), 1331937601)


@pytest.mark.parametrize("tamper", ["body", "url", "hash", "bytes", "missing_manifest"])
def test_author_comment_cache_tampering_is_rejected_without_online_replacement(tmp_path, tamper):
    transport = Mock()
    builder = RepositoryPublicSources(tmp_path / "cache", transport=transport)
    builder.cache_author_comments(offline=True, comment_paths=seed_comments(tmp_path))
    metadata_path = builder.cache_dir / "issuecomment-1331937601.manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tamper == "body":
        document_path = builder.cache_dir / "issuecomment-1331937601.json"
        document_path.write_bytes(comment_document(1331937601, body="changed public body"))
    elif tamper == "missing_manifest":
        metadata_path.unlink()
    else:
        metadata[{"url": "url", "hash": "sha256", "bytes": "bytes"}[tamper]] = "tampered"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="缓存"):
        builder.cache_author_comments(offline=False)
    transport.assert_not_called()


def test_comment_seed_allowlist_and_size_are_enforced(tmp_path):
    paths = seed_comments(tmp_path)
    builder = RepositoryPublicSources(tmp_path / "cache", max_comment_bytes=8, transport=Mock())
    with pytest.raises(ValueError, match="字节上限"):
        builder.cache_author_comments(offline=True, comment_paths=paths)
    with pytest.raises(ValueError, match="白名单"):
        builder.cache_author_comments(offline=True, comment_paths={42: next(iter(paths.values()))})
    with pytest.raises(ValueError, match="白名单"):
        builder.cache_author_comments(offline=True, comment_paths={"1331937601": paths[1331937601]})
    builder.transport.assert_not_called()


def test_comment_download_limit_uses_the_same_controlled_download_layer(tmp_path):
    transport = Mock(return_value=b"x" * 9)
    builder = RepositoryPublicSources(tmp_path / "cache", max_comment_bytes=8, transport=transport)
    with pytest.raises(ValueError, match="字节上限"):
        builder.cache_author_comments(offline=False)
    transport.assert_called_once_with(AUTHOR_COMMENTS[1331937601]["api_url"], max_bytes=8, timeout_s=30)
    assert not (builder.cache_dir / "issuecomment-1331937601.json").exists()

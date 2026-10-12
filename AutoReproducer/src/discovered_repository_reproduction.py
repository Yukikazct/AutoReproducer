"""Run a source-grounded experiment from an uploaded paper's author repository.

No paper identity is registered here. The frozen plan is derived from the actual
PDF and pinned repository, then executed with the repository process owner.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from filelock import FileLock

from src.dependency_cache import AUTO_PREPARATION_LOCK_TIMEOUT
from src.execution_plan import build_plan
from src.method_adapters import digest, read_json, write_json
from src.pdf_input import extract_pdf_input
from src.preset_downloads import atomic_cache_bytes, download_bytes
from src.process_lifecycle import termination_signals
from src.repository_evidence import normalize_repository_url
from src.repository_reproduction import export_repository, materialize_submodules, spec_digest
from src.repository_runner import RepositoryRunner
from src.runtime_preparation import run_owned_process
from src.safety.paths import workspace_path


_MAX_DATA_BYTES = 512 * 1024 * 1024
_REF = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,200}")
_SHA = re.compile(r"[0-9a-f]{40}")


def repository_candidates(document, evidence, resources, *, execution_only=False):
    """Keep declarations first; discovered implementations precede ordinary citations."""
    if evidence['source'] == 'user_supplied':
        return [deepcopy(evidence)]
    links = list(document.repository_links)
    candidates = [*(link for link in links if link.get('is_author_code')), evidence]
    if execution_only:
        discovered = [*(resources.get('alternative_repos') or []),
                      *(url for item in (resources.get('repo_discovery') or {}).get('candidates', [])
                        for url in item.get('repo_urls', []))]
        for raw in discovered:
            url = normalize_repository_url(raw) if isinstance(raw, str) else ''
            if url:
                candidates.append({'url': url, 'source': 'model_discovered', 'page': None,
                    'evidence_type': 'discovered_repository_candidate', 'is_author_code': False,
                    'context': '发现阶段的备用候选；执行模式直接获取源码并尝试原始命令'})
    candidates.extend(links)
    unique, seen = [], set()
    for link in candidates:
        url = link.get('url', '')
        if (url and link.get('evidence_type') != 'reference'
                and link.get('source') in {'pdf_text', 'pdf_annotation', 'user_supplied', 'model_discovered'}
                and url.casefold() not in seen):
            unique.append(deepcopy(link))
            seen.add(url.casefold())
    return unique[:6]


def author_repository_input(request, *, require_author=True):
    """Re-read the actual PDF; preview flags alone never authorize execution."""
    if request.get("mock_mode") or request.get("code") or request.get("corpus_paper"):
        raise ValueError("作者仓库复现需要真实 PDF 输入，不能混用模拟、外部代码或语料预设")
    if not request.get("pdf_path"):
        raise ValueError("缺少用于核验作者代码声明的原始 PDF")
    document = extract_pdf_input(request["pdf_path"])
    expected = (request.get("pdf_input") or {}).get("sha256")
    if expected and expected != document.sha256:
        raise ValueError("PDF 在资源发现后发生变化，不能继续原仓库计划")
    resources = request.get("resources") or {}
    manual = request.get("preferred_repo_url") or ""
    if manual and not require_author:
        selected = normalize_repository_url(manual)
        if not selected:
            raise ValueError("用户提供的 GitHub 仓库地址无效")
        return document, selected, {"url": selected, "raw_url": manual, "page": None,
            "source": "user_supplied", "context": "用户在自动确认失败后补充的仓库地址",
            "evidence_type": "user_supplied_repository", "is_author_code": False}
    selected = normalize_repository_url((resources.get("repo_discovery") or {}).get("selected_repo")
                                        or resources.get("code_repo_url") or resources.get("selected_repo") or "")
    if not selected:
        raise ValueError("没有有效的作者 GitHub 仓库")
    evidence = next((link for link in document.repository_links
                     if (not require_author or (link.get("is_author_code") is True
                         and link.get("evidence_type") == "author_code_statement"))
                     and link.get("evidence_type") != "reference"
                     and link.get("source") in {"pdf_text", "pdf_annotation"}
                     and link["url"].casefold() == selected.casefold()), None)
    if evidence is None:
        if not require_author and request.get('author_execution_only') is True:
            return document, selected, {'url': selected, 'raw_url': selected, 'page': None,
                'source': 'model_discovered', 'context': '资源定位发现的候选；执行模式直接获取源码并尝试原始命令',
                'evidence_type': 'discovered_repository_candidate', 'is_author_code': False}
        raise ValueError("原始 PDF 的作者代码声明没有确认所选仓库" if require_author
                         else "所选仓库没有实际 PDF 中的非参考文献链接证据")
    return document, selected, deepcopy(evidence)


def resolve_repository_revision(url, requested="", *, offline=False):
    """Pin the current public ref exactly; never silently retain another HEAD."""
    requested = str(requested or "")
    if _SHA.fullmatch(requested):
        return requested, {"requested": requested, "resolved_sha": requested, "source": "explicit_commit"}
    if offline:
        raise ValueError("离线作者仓库执行需要预先固定完整 commit SHA")
    ref = requested or "HEAD"
    if not _REF.fullmatch(ref) or ".." in ref or ref.endswith("/"):
        raise ValueError("无效的仓库版本引用")
    patterns = [ref] if ref == "HEAD" or ref.startswith("refs/") else [
        f"refs/heads/{ref}", f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"]
    completed = run_owned_process(["git", "ls-remote", url, *patterns], timeout_s=60)
    if completed.returncode:
        raise RuntimeError("无法从所选作者仓库解析固定版本；未使用其他 HEAD")
    refs = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and _SHA.fullmatch(parts[0]) and parts[1] in patterns:
            refs[parts[1]] = parts[0]
    if f"refs/tags/{ref}^{{}}" in refs:
        refs.pop(f"refs/tags/{ref}", None)
    commits = set(refs.values())
    if len(commits) != 1:
        raise ValueError("作者仓库引用不存在或对应多个不同版本，不能固定本次实验")
    revision = commits.pop()
    return revision, {"requested": requested, "resolved_ref": ref, "resolved_sha": revision,
                      "source": "git_ls_remote"}


def prepare_plan_datasets(root, workspace, plan, snapshot, *, offline=False):
    """Stage only quoted real inputs; bundled data must belong to the Git tree."""
    records = []
    for index, dataset in enumerate(plan.get("datasets", []), 1):
        kind = dataset.get("kind")
        if kind == "bundled":
            files = {}
            for name in dataset.get("paths", []):
                expected = snapshot["files"].get(name)
                path = workspace_path(workspace, name, "bundled dataset", must_exist=True, forbid_git=True)
                if not expected or not path.is_file() or not path.stat().st_size or digest(path) != expected:
                    raise ValueError(f"仓库内数据未通过固定源码校验: {name}")
                files[name] = expected
            if not files:
                raise ValueError("仓库数据声明没有实际文件")
            record = {"kind": kind, "files_sha256": files, "verified": True,
                      "source": snapshot["url"], "revision": snapshot["resolved_sha"]}
            if dataset.get('extraction'):
                from src.bundled_archives import extract_bundled_zip
                record['extracted_files'] = extract_bundled_zip(workspace, dataset['extraction'])
            records.append(record)
        elif kind == "https":
            from urllib.parse import urlsplit
            url, expected = dataset.get("url", ""), dataset.get("sha256", "")
            size = dataset.get("bytes")
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                    or not re.fullmatch(r"[a-f0-9]{64}", expected)
                    or type(size) is not int or not 0 < size <= _MAX_DATA_BYTES):
                raise ValueError("真实数据下载缺少 HTTPS 来源、固定 SHA-256 或合理大小")
            target = workspace_path(workspace, dataset["target"], "downloaded dataset", forbid_git=True)
            if dataset["target"] in snapshot["files"] or target.exists():
                raise ValueError("数据下载不能覆盖作者仓库已有文件")
            cache = workspace_path(root, f"dataset_cache/discovered/{expected}/input", "dataset cache")
            cached = cache.is_file() and cache.stat().st_size == size and digest(cache) == expected
            if not cached:
                if offline:
                    raise RuntimeError("离线缓存缺少本次固定真实数据")
                content = download_bytes(url, max_bytes=size, timeout_s=60)
                if len(content) != size or hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("真实数据下载的完整大小或 SHA-256 不匹配")
                atomic_cache_bytes(cache, content)
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_cache_bytes(target, cache.read_bytes())
            records.append({"kind": kind, "url": url, "canonical_url": url,
                            "download_source": None if cached else url, "cache_hit": cached,
                            "target": dataset["target"], "sha256": expected, "bytes": size,
                            "verified": True})
        elif kind == 'author_loader':
            source = workspace_path(workspace, dataset['path'], 'original data loader', must_exist=True, forbid_git=True)
            if digest(source) != dataset['source_sha256']:
                raise ValueError('原始数据加载器与固定作者源码不一致')
            root_path = workspace_path(workspace, dataset['root'], 'original data root', forbid_git=True)
            if root_path.exists():
                raise ValueError('作者加载器不能覆盖已有数据目录')
            records.append({'kind': kind, 'loader': deepcopy(dataset), 'verified': False, 'pending': True,
                            'reason': '原始作者加载器将在冻结依赖准备完成后下载并校验真实数据'})
        elif kind == 'author_runtime' and plan.get('scope') == 'author_code_execution':
            source = workspace_path(workspace, dataset['path'], 'native dataset source', must_exist=True, forbid_git=True)
            if digest(source) != dataset['source_sha256']:
                raise ValueError('Original native dataset source differs from the frozen plan')
            records.append({'kind': kind, 'loader': deepcopy(dataset), 'verified': False, 'pending': True,
                            'reason': '原作者命令将在实际运行中构造完整真实数据，并记录原始及处理文件'})
        else:
            raise ValueError("本次计划没有受支持的真实数据来源；不会生成合成替代数据")
    if not records and plan.get('scope') != 'author_code_execution':
        raise ValueError("没有完整、可校验的实验数据声明")
    return {"kind": "real" if records else "author_managed_unverified", "sources": records,
            "verified": bool(records) and all(record['verified'] for record in records),
            "sha256": spec_digest(records), "split": "由原文引用冻结的作者实验划分"}


def verify_snapshot(workspace, snapshot, generated):
    for name, expected in {**snapshot["files"], **generated}.items():
        path = workspace_path(workspace, name, "frozen execution source", must_exist=True, forbid_git=True)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"执行后固定源码或适配文件改变: {name}")


def environment_for_plan(plan):
    from src.discovered_repository_plan import (CPU_COMPATIBILITY_PINS, CPU_INDEX_URL,
                                               CUDA_COMPATIBILITY_PINS, CUDA_INDEX_URL, GRAPH_WHEEL_LINKS)
    device = (plan.get("capture") or {}).get("device", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("作者执行设备不在已审阅环境中")
    version_map = CUDA_COMPATIBILITY_PINS if device == "cuda" else CPU_COMPATIBILITY_PINS
    index_url = CUDA_INDEX_URL if device == "cuda" else CPU_INDEX_URL
    requirements = plan.get("requirements") or {}
    pins = requirements.get("compatibility") or []
    checked = []
    for value in pins:
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+-]*)", value)
        if not match:
            raise ValueError("兼容环境必须使用已审阅的精确依赖版本")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        known = {re.sub(r"[-_.]+", "-", key).lower(): version for key, version in version_map.items()}
        expected = str(known.get(name, ""))
        # The public map may store versions or complete requirements.
        expected = expected.split("==")[-1]
        author_pin = (plan.get('scope') == 'author_code_execution' and
                      value in requirements.get('declared_runtime_pins', []) and
                      any(value == line.strip().lstrip('-* ').strip('`')
                          for line in requirements.get('author', [])))
        if match[2] != expected and not author_pin:
            raise ValueError(f"计划依赖不在已审阅兼容环境中: {name}")
        checked.append(value)
    if not checked:
        raise ValueError("缺少冻结的本次执行依赖")
    graph_wheels = any(value.split('==')[0] in {'torch-sparse', 'torch-scatter'} for value in checked)
    wheel_options = f"--find-links {GRAPH_WHEEL_LINKS[device]}\n" if graph_wheels else ''
    return {"requirements_txt": f"--extra-index-url {index_url}\n" + wheel_options + "\n".join(checked) + "\n",
            "author_requirements": deepcopy(requirements.get("author", [])),
            "compatibility_reason": requirements.get("reason", ""),
            "note": f"基于作者依赖与源码导入选择的项目 {device.upper()} 兼容环境；不是论文原始运行环境的等价声明。",
            "auto_prepare": True, "dependency_health_check": True,
            "cache_lock_timeout_s": AUTO_PREPARATION_LOCK_TIMEOUT}


_PREFLIGHT = '''import importlib, importlib.metadata, json, os, platform, sys
from pathlib import Path
root = Path(__file__).resolve().parent
requirements = json.loads((root / "requirements.json").read_text(encoding="utf-8"))
if any(pin.startswith('dgl==') for pin in requirements):
    os.environ['DGLBACKEND'] = 'pytorch'
aliases = {"scikit-learn": "sklearn", "pillow": "PIL", "tensorboardx": "tensorboardX",
           "googledrivedownloader": "google_drive_downloader",
           "torch-geometric": "torch_geometric", "torch-sparse": "torch_sparse",
           "torch-scatter": "torch_scatter", "pyyaml": "yaml", "absl-py": "absl"}
modules, versions = {}, {}
for requirement in requirements:
    name, expected = requirement.split("==", 1)
    actual = importlib.metadata.version(name)
    if actual != expected:
        raise ValueError("Frozen dependency version mismatch: " + name)
    module = importlib.import_module(aliases.get(name.lower(), name.replace("-", "_")))
    modules[name] = str(Path(module.__file__).resolve())
    versions[name] = actual
import torch, numpy
contract = json.loads((root / "config.json").read_text(encoding="utf-8"))["capture"]
device = contract.get("device", "cpu")
if device == "cpu" and torch.cuda.is_available():
    raise ValueError("Generic CPU contract unexpectedly exposes a CUDA device")
if device == "cuda" and not torch.cuda.is_available():
    raise ValueError("Source-required CUDA execution has no usable native GPU")
x = torch.tensor([[1., 2.]], requires_grad=True, device=device)
(x @ x.T).sum().backward()
if not numpy.isfinite(x.grad.detach().cpu().numpy()).all():
    raise ValueError("Native device operation failed")
record = {"versions": versions, "modules": modules, "python": sys.version,
          "platform": platform.platform(), "device": device, "native_probe_pass": True}
if device == "cuda":
    record.update(cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0))
(root / "preflight.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record), flush=True)
'''


class DiscoveredRepositoryReproduction:
    def __init__(self, root, logger, use_docker=False, llm=None):
        self.root, self.logger, self.use_docker, self.llm = Path(root), logger, use_docker, llm
        self.runner = RepositoryRunner(logger=logger)

    @termination_signals()
    def run(self, request, on_event=None):
        from src.run_lifecycle import cancellation_signals
        run_dir = self.root / "runs" / f"repository_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True)
        status = {"status": "running", "pid": os.getpid(), "started_at": time.time()}
        with FileLock(str(run_dir / ".run.lock")), cancellation_signals():
            write_json(run_dir / "run_status.json", status)
            try:
                result = self._run(request, on_event, run_dir)
                status["status"] = "failed" if result.get("error") else "completed"
                return result
            except (KeyboardInterrupt, SystemExit):
                status.update(status="interrupted", reason="作者仓库实验已中断")
                raise
            except Exception:
                status.update(status="failed", reason="作者仓库实验未返回完整结果")
                raise
            finally:
                status["finished_at"] = time.time()
                write_json(run_dir / "run_status.json", status)

    def _run(self, request, on_event, run_dir):
        from src.agents.report_generator import ReportGeneratorAgent
        from src.discovered_repository_plan import build_evidence_packet, bind_full_paper_notes, propose_plan
        started = time.monotonic()
        emit = on_event or (lambda event: None)
        execution_only = request.get('author_execution_only') is True
        phases = [("prepare_repository", "FIND_RESOURCES", "ResourceFinder", "核验论文与 Git 仓库并固定源码版本"),
                  ("select_experiment", "READ_PAPER", "ExperimentProtocol", "判别论文实验方法与源码对应关系"),
                  ("plan_repository", "READ_PAPER", "PaperReader", "依据论文与源码冻结实验计划"),
                  ("prepare_dataset", "FIND_RESOURCES", "ResourceFinder", "准备完整真实数据"),
                  ("prepare_environment", "BUILD_ENV", "EnvBuilder", "准备依赖并进行原生运算检查"),
                  ("execute_repository", "EXECUTE_CODE", "CodeExecutor", "运行固定作者入口与独立评估"),
                  ("verify_protocol", "VALIDATE", "Verifier", "核验固定源码、完成证据与独立指标"),
                  ("generate_report", "GENERATE_REPORT", "ReportGenerator", "生成带来源证据的实验报告")]
        if execution_only:
            phases = [(identifier, state, agent,
                       {'prepare_repository': '获取代码仓库',
                        'select_experiment': '选择原作者完整运行命令',
                        'execute_repository': '运行原作者命令并观察实际训练',
                        'verify_protocol': '核验原作者命令和所属进程已完成'}.get(identifier, title))
                      for identifier, state, agent, title in phases]
            phases.insert(-1, ('compare_author_claims', 'VALIDATE', 'ResultValidator', '对比运行指标与论文宣称'))
        emit({"type": "pipeline_plan", "pipeline": "repository", "stages": [
            {"id": identifier, "agent": agent, "title": title, "status": "waiting"}
            for identifier, _, agent, title in phases]})
        def event(identifier, status, reason=""):
            _, state, agent, title = next(item for item in phases if item[0] == identifier)
            self.logger.log(agent, identifier, status.upper(), reason or title)
            if status == "running":
                self.logger.begin_plan(identifier)
            else:
                self.logger.end_plan(identifier)
            emit({"type": "state", "state": state, "agent": agent, "phase_id": identifier,
                  "status": status, "reason": reason or title})
        data = {"run_dir": str(run_dir.resolve()), "report_path": str((run_dir / "report.md").resolve()),
                "paper_info": deepcopy(request.get("paper_info") or {}), "resources": deepcopy(request.get("resources") or {}),
                "verifications": deepcopy(request.get("verifications") or []),
                "fix_records": deepcopy(request.get("fix_records") or []), "repository_executed": False,
                "reproduction_scope": "selected_paper_experiment", "execution_source": "author_repository",
                "optimization": {"mode": "off", "available": False, "optimized": False,
                                 "status": "disabled", "reason": "运行已冻结作者实验协议"}}
        if execution_only:
            data['reproduction_scope'] = 'author_code_execution'
        for key in ('reading_coverage', 'reading_batches', 'reading_originals'):
            if key in request:
                data[key] = deepcopy(request[key])
        phase, error = "prepare_repository", None
        before_calls = self.llm.get_call_count() if self.llm else 0
        try:
            event(phase, "running")
            if self.use_docker:
                raise ValueError("通用作者仓库执行当前需要本地受控进程模式")
            from src.agents.repository_verifier import RepositoryVerifierAgent
            document, url, evidence = author_repository_input(request, require_author=False)
            manual = evidence["source"] == "user_supplied"
            if manual:
                data["execution_source"] = "user_supplied_repository"
                data["resources"].update(code_repo_url=url, selected_repo=url, selection_evidence=deepcopy(evidence),
                    repository_identity={"status": "user_supplied_unconfirmed", "source": "user_supplied_pinned_git_llm_review",
                        "author_identity_confirmed": False, "reason": "用户补充的仓库，论文与实现关系尚未核验", "evidence": deepcopy(evidence)})
            data["pdf_input"] = {"sha256": document.sha256, "pages": document.page_count,
                                 "bytes": document.byte_size, "readable": True}
            pin = (data["resources"].get("repo_discovery") or {}).get("pinned_revision", "")
            offline = bool(request.get("offline"))
            unique = repository_candidates(document, evidence, data['resources'], execution_only=execution_only)
            attempts, packet, review = [], None, None
            for index, candidate in enumerate(unique):
                candidate_url = candidate["url"]
                attempt = {"repository_url": candidate_url, "accepted": False}
                attempts.append(attempt)
                try:
                    revision, pin_evidence = resolve_repository_revision(
                        candidate_url, pin if candidate_url.casefold() == url.casefold() else "", offline=offline)
                    workspace = run_dir / ("repo" if index == 0 else f"candidate_repos/{index + 1}")
                    snapshot = export_repository(self.root, {"repository": {"url": candidate_url, "revision": revision}},
                                                 workspace, offline=offline, required_files=[], include_submodules=False)
                    if snapshot.get("resolved_sha") != revision or snapshot.get("url") != candidate_url:
                        raise ValueError("导出的仓库与所选作者来源或固定版本不一致")
                    attempt.update(revision=revision, archive_sha256=snapshot.get("archive_sha256"),
                                   workspace=str(workspace.resolve()))
                    # Allow discovered/user-provided URLs without a literal
                    # author-code statement in the PDF. This flag only controls
                    # evidence acquisition; execution mode skips the reviewer.
                    packet = build_evidence_packet(request["pdf_path"], workspace, snapshot, for_identity_review=True)
                    if execution_only:
                        packet['author_execution_only'] = True
                    bind_full_paper_notes(packet, request)
                    if manual:
                        packet["repository_input"] = {"source": "user_supplied", "url": url}
                    elif candidate.get('source') == 'model_discovered':
                        packet['repository_input'] = {'source': 'model_discovered', 'url': candidate_url}
                    if packet["pdf"]["sha256"] != document.sha256:
                        raise ValueError("原始 PDF 在仓库准备期间发生变化，不能使用不同论文继续规划")
                    write_json(run_dir / f"repository_identity_packet_{index + 1}.json", packet)
                    if execution_only:
                        review = {'status': 'skipped', 'author_identity_confirmed': False,
                            'execution_allowed': True,
                            'reason': '按执行目标直接获取仓库；未进行 LLM 论文与仓库匹配审核。'}
                    else:
                        review = RepositoryVerifierAgent(self.llm, self.logger).run(packet)
                    if snapshot.get('submodules_pending'):
                        materialize_submodules(self.root, snapshot, offline=offline)
                        packet = build_evidence_packet(request['pdf_path'], workspace, snapshot, for_identity_review=True)
                        packet['author_execution_only'] = execution_only
                        bind_full_paper_notes(packet, request)
                        packet['repository_input'] = {'source': candidate['source'], 'url': candidate_url}
                    attempt.update(accepted=True, review=review, revision=revision)
                    url, evidence = candidate_url, deepcopy(candidate)
                    packet["repository_identity_review"] = review
                    packet["repository_identity_pending"] = False
                    break
                except Exception as exc:
                    review = None
                    attempt.update(error=str(exc), review=getattr(exc, "repository_review", None))
                    if digest(Path(request["pdf_path"])) != document.sha256:
                        raise ValueError("原始 PDF 在仓库准备期间发生变化，不能继续") from exc
                finally:
                    write_json(run_dir / "repository_identity_attempts.json", attempts)
            data["repository_identity_attempts"] = attempts
            if review is None:
                raise ValueError(("无法获取候选仓库源码。" if execution_only else
                                 "用户提供的仓库与论文实现内容未确认匹配；未启动实验。" if manual
                                 else "候选仓库与论文的作者实现关系未确认；未启动实验。")
                                 + "；".join(attempt.get("error", "") for attempt in attempts))
            evidence["semantic_review"] = review
            data["resources"].update(code_repo_url=url, selected_repo=url, selection_evidence=evidence,
                repository_identity={"status": ("unreviewed_execution" if execution_only else
                                         "user_supplied_verified" if manual else "author_verified"
                                         if review['author_identity_confirmed'] else "matching_implementation_verified"),
                                     "source": ("pinned_git_direct_execution" if execution_only else
                                         "user_supplied_pinned_git_llm_review" if manual else
                                         "discovered_pinned_git_llm_review" if evidence.get('source') == 'model_discovered'
                                         else "pdf_pinned_git_llm_review"),
                                     "author_identity_confirmed": review["author_identity_confirmed"],
                                     "reason": review["reason"], "evidence": deepcopy(evidence), "review": review})
            data["resources"].setdefault("repo_discovery", {}).update(
                selected_repo=url, pinned_revision=revision, repository_identity=data["resources"]["repository_identity"])
            data.update(repository=snapshot, revision_resolution=pin_evidence)
            write_json(run_dir / "repository.json", snapshot)
            event(phase, "success")
            phase = "select_experiment"; event(phase, "running")
            from src.agents.experiment_protocol import ExperimentProtocolAgent
            try:
                protocol = (None if execution_only else
                            ExperimentProtocolAgent(self.llm, self.logger).run(packet, workspace))
            except BaseException:
                try:
                    write_json(run_dir / 'source_packet.json', packet)
                except Exception as persistence_error:
                    data['source_packet_persistence_error'] = str(persistence_error)
                raise
            if protocol is not None:
                data['experiment_protocol'] = protocol
                packet['experiment_protocol'] = protocol
                write_json(run_dir / 'experiment_protocol.json', protocol)
                event(phase, "success")
            else:
                event(phase, 'skipped', '执行目标为先运行作者代码；完整命令与真实数据由后续独立规划审查核验')
            phase = "plan_repository"; event(phase, "running")
            if digest(Path(request["pdf_path"])) != document.sha256:
                raise ValueError("原始 PDF 在仓库准备期间发生变化，不能使用不同论文继续规划")
            write_json(run_dir / "source_packet.json", packet)
            planning_error = None
            try:
                if execution_only:
                    from src.author_execution_plan import propose_author_execution_plan
                    plan = propose_author_execution_plan(self.llm, packet, workspace)
                else:
                    plan = propose_plan(self.llm, packet, workspace)
            except BaseException as exc:
                planning_error = exc
                raise
            finally:
                # A rejected proposal may already have requested more source.
                # Keep that exact packet for audit without replacing the
                # original rejection if persisting diagnostic evidence fails.
                try:
                    write_json(run_dir / "source_packet.json", packet)
                except Exception as persistence_error:
                    if planning_error is None:
                        raise
                    data["source_packet_persistence_error"] = str(persistence_error)
            plan_review = plan.get('semantic_review') or {}
            if plan_review.get('accepted') is not True and not (
                    execution_only and plan_review.get('advisory') is True and plan_review.get('execution_allowed') is True):
                raise ValueError("独立的论文与作者协议审查尚未通过，不能开始训练")
            data["grounded_repository_plan"] = plan
            data["spec_sha256"] = spec_digest(plan)
            write_json(run_dir / "experiment_spec.json", {"spec": plan, "sha256": data["spec_sha256"]})
            data["paper_info"]["metrics"] = {item["name"]: item["reference"] for item in plan["metrics"]}
            data["paper_info"]["required_metrics"] = [item["name"] for item in plan["metrics"]]
            environment = environment_for_plan(plan)
            environment["offline"] = offline
            data["env_config"] = deepcopy(environment)
            data["experiment_spec"] = {"id": "discovered_author_experiment", "adapter_id": "discovered",
                "label": "依据本次 PDF 与用户提供的固定仓库生成的实验计划" if manual else "依据本次 PDF 与固定作者源码生成的实验计划",
                "repository": plan["repository"],
                "paper": deepcopy(data["paper_info"]), "parameters": plan.get("parameters", {}),
                "environment": deepcopy(environment), "validation": {"scope": "selected_paper_experiment",
                    "note": "本次论文与作者源码引文确定实验范围；独立指标与完整执行证据共同决定结果。"}}
            if execution_only:
                data['experiment_spec']['validation'] = {'scope': 'author_code_execution',
                    'note': '核验原作者完整命令的实际运行和进程结束；尚未验收论文数值。'}
            event(phase, "success")
            phase = "prepare_dataset"; event(phase, "running")
            datasets = prepare_plan_datasets(self.root, workspace, plan, snapshot, offline=offline)
            data["dataset_provenance"] = datasets
            write_json(run_dir / "dataset.json", datasets)
            event(phase, 'success' if datasets['verified'] else 'waiting',
                  '' if datasets['verified'] else '等待固定依赖就绪后执行原始作者数据加载器')
            if execution_only and not plan['datasets']:
                event(phase, 'skipped', '输入数据由原作者程序加载；完整数据来源尚未核验，先验证代码可运行')
            self._execute_frozen(request, data, plan, snapshot, workspace, run_dir, environment, emit, event)
        except (KeyboardInterrupt, SystemExit) as exc:
            phase = getattr(exc, "reproduction_phase", phase)
            data["interrupted"] = True
            if getattr(exc, "execution", None) is not None:
                data["environment_preparation" if phase == "prepare_environment" else "execution"] = exc.execution
            data["validation"] = {"status": "interrupted", "is_reproduced": None, "result_level": "failed",
                                  "reason": "实验已中断，未完成的训练和指标不能当作成功"}
            event(phase, "interrupted", data["validation"]["reason"])
            write_json(run_dir / "result.json", {"state": "ERROR", "data": data, "error": "interrupted"})
            raise
        except Exception as exc:
            phase = getattr(exc, "reproduction_phase", phase)
            error = str(exc) or type(exc).__name__
            if phase == "prepare_repository":
                try:
                    fallback_pdf = extract_pdf_input(request["pdf_path"])
                    data["pdf_input"] = {"sha256": fallback_pdf.sha256, "pages": fallback_pdf.page_count,
                                         "bytes": fallback_pdf.byte_size, "readable": True}
                    data["repository_fallback"] = {"available": True, "pdf_sha256": fallback_pdf.sha256,
                        "reason": error, "action": "用户可补充 GitHub 仓库地址，重新核对论文与仓库后重试"}
                except (ValueError, OSError, KeyError):
                    pass
            if getattr(exc, "planning_evidence", None) is not None:
                data["plan_rejection"] = deepcopy(exc.planning_evidence)
                write_json(run_dir / "plan_rejection.json", data["plan_rejection"])
            data["validation"] = {"status": "execution_failed" if phase in {"execute_repository", "verify_protocol"}
                                  else "insufficient_evidence", "failure_phase": phase, "is_reproduced": None,
                                  "failure_kind": getattr(exc, 'category', 'execution_failure' if phase in {'execute_repository','verify_protocol'}
                                                          else 'runtime_error'),
                                  "result_level": "failed", "reason": error}
            data.setdefault("execution", {"mode": "repository", "executed": False, "success": False,
                                          "repository_executed": False})
            event(phase, "error", error)
        data["total_llm_calls"] = (self.llm.get_call_count() if self.llm else 0) - before_calls
        self.logger.add_llm_calls(data["total_llm_calls"])
        data["audit_stats"] = self.logger.get_stats()
        data["run_elapsed_s"] = time.monotonic() - started
        event("generate_report", "running")
        try:
            data["report"] = ReportGeneratorAgent(self.logger).run(data, report_path=data["report_path"])["report"]
            (run_dir / "report.md").write_text(data["report"], encoding="utf-8")
            result = {"state": "ERROR" if error else "COMPLETED", "data": data, "error": error}
            write_json(run_dir / "result.json", result)
            event("generate_report", "success")
            return result
        except (KeyboardInterrupt, SystemExit) as exc:
            data["interrupted"] = True
            data["report_error"] = {"phase": "generate_report", "reason": str(exc) or "报告生成已中断"}
            event("generate_report", "error", "报告生成已中断")
            try:
                write_json(run_dir / "result.json", {"state": "ERROR", "data": data, "error": "interrupted"})
            except Exception:
                # Preserve the cancellation even when diagnostic persistence
                # is unavailable; completed training evidence stays in memory.
                pass
            raise
        except Exception as exc:
            error = "报告生成失败：" + str(exc)
            data["report_error"] = {"phase": "generate_report", "reason": str(exc)}
            event("generate_report", "error", error)
            result = {"state": "ERROR", "data": data, "error": error}
            try:
                write_json(run_dir / "result.json", result)
            except OSError:
                pass
            return result

    def _execute_frozen(self, request, data, plan, snapshot, workspace, run_dir, environment, emit, event):
        """Materialize project capture utilities, then run and independently check."""
        from src.discovered_repository_runtime import CAPTURE_DIR, materialize_capture, verify_capture
        phase = "prepare_environment"
        try:
            event(phase, "running")
            execution_only = plan.get('scope') == 'author_code_execution'
            review = plan.get('semantic_review') or {}
            if (review.get('accepted') is not True and not (
                    execution_only and review.get('advisory') is True and review.get('execution_allowed') is True)):
                raise ValueError("独立协议审查未通过")
            if read_json(run_dir / "experiment_spec.json") != {"spec": plan, "sha256": data["spec_sha256"]}:
                raise ValueError("落盘实验协议与冻结的本次计划不一致")
            packet = read_json(run_dir / "source_packet.json")
            canonical = json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != plan["evidence_sha256"]:
                raise ValueError("落盘来源包与冻结的实验依据不一致")
            if execution_only:
                from src.author_execution_plan import materialize_author_execution
                materialized = materialize_author_execution(workspace, plan, data['spec_sha256'], snapshot)
            else:
                materialized = materialize_capture(workspace, plan, data["spec_sha256"], snapshot=snapshot)
            generated = materialized["files"]
            runtime_dir = workspace / CAPTURE_DIR
            (runtime_dir / "preflight.py").write_text(_PREFLIGHT, encoding="utf-8")
            write_json(runtime_dir / "requirements.json", plan["requirements"]["compatibility"])
            for name in ("preflight.py", "requirements.json"):
                generated[f"{CAPTURE_DIR}/{name}"] = digest(runtime_dir / name)
            loaders = [item for item in plan['datasets'] if item['kind'] == 'author_loader']
            runtime_loaders = [item for item in plan['datasets'] if item['kind'] == 'author_runtime']
            if loaders:
                preparation_source = Path(__file__).parent / 'experiments/author_dataset_prepare.py'
                (runtime_dir / 'prepare_datasets.py').write_bytes(preparation_source.read_bytes())
                write_json(runtime_dir / 'dataset_loaders.json', {'loaders': loaders,
                    'source_files': snapshot['files'], 'offline': bool(request.get('offline'))})
                for name in ('prepare_datasets.py', 'dataset_loaders.json'):
                    generated[f'{CAPTURE_DIR}/{name}'] = digest(runtime_dir / name)
            write_json(run_dir / "adapter_manifest.json", {"files": generated,
                       "note": "原样保留作者源码；项目包装器仅记录训练、导出模型及独立评估"})
            verify_snapshot(workspace, snapshot, generated)
            steps = materialized["steps"]
            execution_plan = build_plan(profile="discovered_author_experiment", spec_sha256=data["spec_sha256"],
                workspace=workspace, repository=plan["repository"], dataset=data["dataset_provenance"], steps=steps)
            data["execution_plan"] = execution_plan
            write_json(run_dir / "execution_plan.json", execution_plan)
            write_json(run_dir / "environment.json", environment)
            def verify_frozen_inputs():
                if digest(request["pdf_path"]) != plan["pdf_sha256"]:
                    raise ValueError("原始 PDF 在执行准备或运行期间发生变化")
                if read_json(run_dir / "experiment_spec.json") != {"spec": plan, "sha256": data["spec_sha256"]}:
                    raise ValueError("落盘实验协议在执行期间发生变化")
                if read_json(run_dir / "execution_plan.json") != execution_plan:
                    raise ValueError("落盘执行计划在执行期间发生变化")
                stored_packet = read_json(run_dir / "source_packet.json")
                encoded = json.dumps(stored_packet, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")
                if hashlib.sha256(encoded).hexdigest() != plan["evidence_sha256"]:
                    raise ValueError("落盘来源包在执行期间发生变化")
                verify_snapshot(workspace, snapshot, generated)
                for dataset in data["dataset_provenance"]["sources"]:
                    for name, record in dataset.get('extracted_files', {}).items():
                        path = workspace_path(workspace, name, 'frozen archive data', must_exist=True, forbid_git=True)
                        if digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
                            raise ValueError('作者压缩包中的真实数据在执行期间发生变化')
                    if dataset["kind"] == "https":
                        path = workspace_path(workspace, dataset["target"], "frozen real data", must_exist=True)
                        if digest(path) != dataset["sha256"] or path.stat().st_size != dataset["bytes"]:
                            raise ValueError("真实下载数据在执行期间发生变化")
                    elif dataset['kind'] == 'author_loader' and dataset.get('verified') is True:
                        for name, record in dataset['files'].items():
                            path = workspace_path(workspace, name, 'frozen author data', must_exist=True, forbid_git=True)
                            if digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
                                raise ValueError('作者加载器下载或处理的真实数据在执行期间发生变化')
            verify_frozen_inputs()
            if request.get("prepare_only"):
                data["execution"] = {"mode": "repository", "executed": False, "success": False,
                                     "repository_executed": False, "reason": "仅准备计划，尚未执行作者训练"}
                data["validation"] = {"status": "prepared", "is_reproduced": None,
                    "result_level": "prepared", "reason": ('源码与协议已冻结；作者加载器数据仍待依赖准备后下载，尚未执行训练'
                    if loaders else '源码和完整原命令已冻结；实际数据将在作者执行中构造，尚未训练'
                    if runtime_loaders else '源码、完整真实数据与协议已冻结，尚未执行训练')}
                event(phase, "success", "执行计划已准备，未安装或训练")
                if loaders or runtime_loaders:
                    event('prepare_dataset', 'skipped', '仅准备计划；作者加载器尚未下载真实数据')
                for identifier in ("execute_repository", "verify_protocol"):
                    event(identifier, "skipped", "仅准备模式")
                if execution_only:
                    event('compare_author_claims', 'skipped', '尚未运行作者代码')
                return

            def progress(item):
                emit({**item, "phase_id": phase})

            def repair_numeric(result):
                # Only an actual failed author observer can propose this one
                # deterministic repair. Ordinary exceptions remain failures.
                if 'Integers to negative integer powers are not allowed.' not in str(
                        (result.get('final') or {}).get('stderr', '')):
                    return None
                for failed in result.get('steps', []):
                    if failed.get('executed') is not True or failed.get('exit_code') in (None, 0):
                        continue
                    identifier = failed.get('id')
                    if not any(step['id'] == identifier for step in plan['steps']):
                        continue
                    candidate_path = runtime_dir / ('numeric_failure_' + identifier + '.json')
                    if not candidate_path.exists():
                        continue
                    candidate = read_json(candidate_path)
                    fix = candidate.get('repair') or {}
                    if (candidate.get('spec_sha256') != data['spec_sha256']
                            or candidate.get('step') != identifier or candidate.get('optimizer_steps') != 0
                            or fix.get('kind') != 'numpy_float_reciprocal'
                            or fix.get('source_sha256') != snapshot['files'].get(fix.get('path'))):
                        return None
                    from src.experiments.author_numeric_compatibility import power_call
                    import ast
                    source = workspace_path(workspace, fix['path'], 'numeric compatibility source', must_exist=True)
                    call = power_call(ast.parse(source.read_bytes()), fix.get('line'), fix.get('column'))
                    if (call is None or ast.get_source_segment(source.read_text('utf-8'), call) != fix.get('original_expression')
                            or fix.get('replacement') != call.func.value.id + '.float_power'):
                        return None
                    capture_config = read_json(runtime_dir / 'config.json')
                    configured = capture_config.setdefault('numeric_compatibility', [])
                    if any(item['path'] == fix['path'] and item['line'] == fix['line']
                           and item['column'] == fix['column'] for item in configured):
                        return None
                    verify_frozen_inputs()
                    configured.append(fix)
                    write_json(runtime_dir / 'config.json', capture_config)
                    generated[f'{CAPTURE_DIR}/config.json'] = digest(runtime_dir / 'config.json')
                    write_json(run_dir / 'adapter_manifest.json', {'files': generated,
                        'note': '原仓库文件不变；记录实际失败后的指定倒数表达式浮点兼容修复，执行时仅转换该 AST 调用。'})
                    return fix
                return None

            def run_steps(selected, runtime_environment):
                if execution_only:
                    from src.author_runtime_dependencies import run_with_dependency_repair
                    return run_with_dependency_repair(self.runner, workspace, selected, runtime_environment,
                        plan, packet, run_dir, data, progress, verify_frozen_inputs, repair_numeric)
                return self.runner.run(workspace, selected, runtime_environment, on_event=progress)

            preflight = [{"id": "native_preflight", "kind": "check", "required": True,
                          "argv": ["python", "-u", f"{CAPTURE_DIR}/preflight.py"], "cwd": ".",
                          "timeout_s": 180, "env": {"CUDA_VISIBLE_DEVICES": ""}
                              if plan["capture"].get("device", "cpu") == "cpu" else {},
                          "artifacts": [f"{CAPTURE_DIR}/preflight.json"]}]
            preparation = run_steps(preflight, environment)
            data["environment_preparation"] = preparation
            write_json(run_dir / "environment_preparation.json", preparation)
            if preparation.get("success") is not True:
                raise RuntimeError("依赖准备或原生导入/运算检查失败：" + str(
                    preparation.get("reason") or preparation.get("final", {}).get("stderr", ""))[-2000:])
            environment = preparation.get('effective_env_config') or environment
            probe = read_json(runtime_dir / "preflight.json")
            if probe.get("native_probe_pass") is not True or probe.get("device") != plan["capture"].get("device", "cpu"):
                raise ValueError("原生设备运算检查缺少成功证据")
            data["native_preflight"] = probe
            verify_frozen_inputs()
            event(phase, "success")
            if loaders:
                phase = 'prepare_dataset'; event(phase, 'running')
                dataset_step = [{'id': 'prepare_author_datasets', 'kind': 'check', 'required': True,
                    'argv': ['python', '-u', f'{CAPTURE_DIR}/prepare_datasets.py'], 'cwd': '.', 'timeout_s': 600,
                    'env': {'CUDA_VISIBLE_DEVICES': ''}, 'artifacts': [f'{CAPTURE_DIR}/prepared_datasets.json']}]
                dataset_execution = run_steps(dataset_step,
                    {**environment, 'auto_prepare': False, 'require_prepared': True})
                data['dataset_preparation_execution'] = dataset_execution
                write_json(run_dir / 'dataset_preparation_execution.json', dataset_execution)
                if dataset_execution.get('success') is not True:
                    raise RuntimeError('原始作者数据准备未完整结束：' + str(dataset_execution.get('reason') or
                        dataset_execution.get('final', {}).get('stderr', ''))[-2000:])
                environment = dataset_execution.get('effective_env_config') or environment
                prepared = read_json(runtime_dir / 'prepared_datasets.json')
                records = prepared.get('sources')
                if prepared.get('verified') is not True or not isinstance(records, list) or len(records) != len(loaders):
                    raise ValueError('原始作者数据准备缺少全部加载器的完成证据')
                for record, loader in zip(records, loaders):
                    if record.get('verified') is not True or record.get('loader') != loader or not record.get('files'):
                        raise ValueError('真实数据准备记录与冻结的原始加载器不一致')
                    for name, value in record['files'].items():
                        target = workspace_path(workspace, name, 'prepared author data', must_exist=True, forbid_git=True)
                        if (not name.startswith(loader['root'] + '/') or name in snapshot['files']
                                or type(value.get('bytes')) is not int or value['bytes'] <= 0
                                or target.stat().st_size != value['bytes'] or digest(target) != value.get('sha256')):
                            raise ValueError('作者加载器的真实文件在训练前未通过完整校验')
                iterator = iter(records)
                provenance = data['dataset_provenance']
                provenance['sources'] = [next(iterator) if item['kind'] == 'author_loader' else item
                                         for item in provenance['sources']]
                provenance.update(verified=all(item.get('verified') is True for item in provenance['sources']),
                                  sha256=spec_digest(provenance['sources']))
                write_json(run_dir / 'dataset.json', provenance)
                capture_configuration = read_json(runtime_dir / 'config.json')
                capture_configuration['data_files'] = {name: record for source in records for name, record in source['files'].items()}
                write_json(runtime_dir / 'config.json', capture_configuration)
                for name in ('config.json', 'prepared_datasets.json'):
                    generated[f'{CAPTURE_DIR}/{name}'] = digest(runtime_dir / name)
                write_json(run_dir / 'adapter_manifest.json', {'files': generated,
                    'note': '保留作者源码；数据由原始加载器下载并在训练前冻结，包装器仅捕获与独立评估'})
                execution_plan = build_plan(profile='discovered_author_experiment', spec_sha256=data['spec_sha256'],
                    workspace=workspace, repository=plan['repository'], dataset=provenance, steps=steps)
                data['execution_plan'] = execution_plan
                write_json(run_dir / 'execution_plan.json', execution_plan)
                verify_frozen_inputs()
                event(phase, 'success' if provenance['verified'] else 'waiting',
                      '' if provenance['verified'] else '等待原作者命令构造其余完整真实数据')
            native_pending = execution_only and runtime_loaders and all(
                item.get('verified') is True or item['kind'] == 'author_runtime'
                for item in data['dataset_provenance']['sources'])
            author_managed_data = execution_only and not plan['datasets']
            if data['dataset_provenance'].get('verified') is not True and not native_pending and not author_managed_data:
                raise ValueError('完整真实数据尚未冻结，不能开始作者训练')
            if request.get("prepare_environment"):
                data["execution"] = {"mode": "repository", "executed": False, "success": False,
                                     "repository_executed": False, "reason": "环境已准备，尚未执行作者训练"}
                data["validation"] = {"status": "environment_ready", "is_reproduced": None,
                                      "result_level": "prepared", "reason": ('原生环境已准备；原作者命令尚未构造实际数据或开始训练'
                                      if native_pending else '真实数据、协议和原生环境已准备，尚未执行训练')}
                if runtime_loaders:
                    event('prepare_dataset', 'skipped', '仅准备环境；实际数据构造和训练尚未执行')
                for identifier in ("execute_repository", "verify_protocol"):
                    event(identifier, "skipped", "仅准备实验环境")
                if execution_only:
                    event('compare_author_claims', 'skipped', '尚未运行作者代码')
                return
            phase = "execute_repository"; event(phase, "running")
            verify_frozen_inputs()
            # Preparation does not consume the separately frozen author runtime
            # deadline. The fresh evaluator has its own bounded 300-second step.
            prepared_environment = {**environment, "auto_prepare": False, "require_prepared": True,
                "deadline_monotonic": time.monotonic() + sum(step["timeout_s"] for step in steps) + 2}
            execution = run_steps(execution_plan, prepared_environment)
            data["execution"] = execution
            training_id = next(step["id"] for step in plan["steps"] if step["kind"] == "train")
            data["repository_executed"] = any(step.get("id") == training_id and step.get("executed")
                                               for step in execution.get("steps", []))
            execution["repository_executed"] = data["repository_executed"]
            execution["artifacts"] = [item for item in execution.get("artifacts", [])
                                       if item.get("name") not in snapshot["files"]]
            execution.setdefault("final", {})["artifacts"] = execution["artifacts"]
            write_json(run_dir / "execution.json", execution)
            if execution.get("success") is not True:
                raise RuntimeError("作者训练或独立评估未完整结束：" + str(execution.get("reason") or
                    execution.get("final", {}).get("stderr", ""))[-2000:])
            event(phase, "success")
            phase = "verify_protocol"; event(phase, "running")
            verify_frozen_inputs()
            if execution_only:
                from src.author_execution_plan import verify_author_execution
                validation = verify_author_execution(plan, workspace, execution, data['spec_sha256'])
                if runtime_loaders:
                    from src.runtime_dataset_evidence import verify_runtime_datasets
                    records = verify_runtime_datasets(plan, validation['author_execution'], workspace)
                    iterator = iter(records)
                    provenance = data['dataset_provenance']
                    provenance['sources'] = [next(iterator) if item['kind'] == 'author_runtime' else item
                                             for item in provenance['sources']]
                    provenance.update(verified=all(item.get('verified') is True for item in provenance['sources']),
                                      sha256=spec_digest(provenance['sources']))
                    write_json(run_dir / 'dataset.json', provenance)
                    event('prepare_dataset', 'success', '原作者数据构造、完整文件及训练期间数据不变已核验')
                if data['dataset_provenance'].get('verified') is not True and not author_managed_data:
                    raise ValueError('Original native dataset has not completed real-data verification')
                validation['completion_checks'].update(frozen_sources_verified=True, native_environment_verified=True)
                validation['dataset_provenance_verified'] = data['dataset_provenance'].get('verified') is True
                if validation.get('numeric_compatibility'):
                    validation.setdefault('warnings', []).append('执行包含实际报错后记录的浮点倒数兼容修复；原仓库文件未改写。')
                identity = data['resources']['repository_identity']
                if identity.get('status') == 'unreviewed_execution':
                    validation.setdefault('warnings', []).append('本次直接运行所选仓库；论文与实现关系、作者归属未审核。')
                elif not identity.get('author_identity_confirmed'):
                    validation.setdefault('warnings', []).append('仓库实现与论文匹配；作者归属尚未确认。')
                if not author_managed_data:
                    validation['completion_checks']['real_data_frozen'] = True
                else:
                    validation.setdefault('warnings', []).append('作者程序已运行；完整数据来源与规模尚未核验。')
                data['validation'] = validation
                execution['author_execution_verification'] = validation
                write_json(run_dir / 'author_execution_verification.json', validation)
                write_json(run_dir / 'execution.json', execution)
                event(phase, 'success', validation['reason'])
                phase = 'compare_author_claims'; event(phase, 'running')
                from src.author_claim_comparison import compare_author_claims
                comparison = compare_author_claims(self.llm, read_json(run_dir / 'source_packet.json'),
                                                    plan, workspace, execution)
                validation['author_claim_comparison'] = comparison
                validation['metric_records'] = comparison['metric_records']
                write_json(run_dir / 'author_claim_comparison.json', comparison)
                write_json(run_dir / 'author_execution_verification.json', validation)
                write_json(run_dir / 'execution.json', execution)
                event(phase, 'success' if comparison['comparisons'] else 'skipped', comparison['reason'])
                return
            validation = verify_capture(plan, workspace, execution, data["spec_sha256"])
            data["validation"] = validation
            execution["protocol_verification"] = {"pass": True, **validation["capture"]}
            execution["independent_metrics"] = validation["independent_metrics"]
            write_json(run_dir / "protocol_verification.json", execution["protocol_verification"])
            write_json(run_dir / "independent_metrics.json", execution["independent_metrics"])
            write_json(run_dir / "metrics.json", {"records": validation["comparisons"],
                       "spec_sha256": data["spec_sha256"]})
            write_json(run_dir / "execution.json", execution)
            event(phase, "success", validation["reason"])
        except BaseException as exc:
            exc.reproduction_phase = phase
            raise

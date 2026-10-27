"""Run a source-grounded experiment from an uploaded paper's author repository.

No paper identity is registered here. The frozen plan is derived from the actual
PDF and pinned repository, then executed with the repository process owner.
"""
from copy import deepcopy
from contextlib import contextmanager, nullcontext
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
from src.environment_lifecycle import bind_environment_run, release_run_environment
from src.intelligent_recovery import current_investigation, investigating_run, InvestigationStopped


_MAX_DATA_BYTES = 512 * 1024 * 1024
_REF = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,200}")
_SHA = re.compile(r"[0-9a-f]{40}")


def _diagnose_outcome(llm, plan, packet, workspace, execution, validation, run_dir, data, emit):
    """Keep analysis errors informational; propagate process safety stops."""
    from src.result_diagnosis import diagnose_experiment_result
    try:
        repaired = data.get('source_repair') or {}
        if repaired.get('status') == 'succeeded':
            # Numerical output belongs to the approved repair copy. Do not
            # explain it using the original, now different source snapshot.
            workspace = Path(repaired['workspace'])
            inventory = packet.get('file_inventory') or {}
            packet = {**packet, 'file_inventory': {
                name: {**inventory.get(name, {}), 'sha256': sha}
                for name, sha in repaired['effective_source_files'].items()}}
        investigation = current_investigation()
        diagnosis = diagnose_experiment_result(investigation.wrap_llm(llm) if investigation else llm,
            plan, packet, workspace, execution, validation, run_dir, data, emit)
        if investigation and diagnosis:
            investigation.investigate(trigger='result_discrepancy', plan=plan, packet=packet,
                workspace=workspace, environment=execution.get('effective_env_config') or data.get('env_config') or {},
                execution=execution, validation=validation)
            investigation.ensure_running()
        return diagnosis
    except InvestigationStopped:
        raise
    except Exception as exc:
        # Keep raw exception text out of the artifact: a provider may include
        # credentials or submitted source in its error message.
        data['outcome_diagnosis'] = {
            'version': 1, 'status': 'unavailable',
            'reason': '辅助结果诊断暂不可用；已完成的作者执行与核验结论保留。',
            'error_type': type(exc).__name__,
            'actions': {'mutated': False, 'retrained': False},
        }
        try:
            write_json(run_dir / 'outcome_diagnosis.json', data['outcome_diagnosis'])
        except OSError:
            pass
        return data['outcome_diagnosis']


@contextmanager
def _frozen_input_validation():
    """A changed frozen input must never become an environment fallback."""
    try:
        yield
    except (ValueError, OSError, TypeError, KeyError) as exc:
        exc.category = 'source_changed'
        raise


def repository_candidates(document, evidence, resources, *, execution_only=False):
    """Keep declarations first; discovered implementations precede ordinary citations."""
    if evidence['source'] == 'user_supplied':
        return [deepcopy(evidence)]
    links = list(document.repository_links)
    candidates = [*(link for link in links if link.get('is_author_code')), evidence]
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


def linked_training_repositories(packet):
    """Follow only README links explicitly describing the same method's code/training."""
    candidates, seen = [], {packet['repository']['url'].casefold()}
    for path, source in packet['repository']['files'].items():
        if not Path(path).name.lower().startswith('readme'):
            continue
        for line in source['text'].splitlines():
            lower = line.lower()
            if (not any(word in lower for word in ('python version', 'training', 'train ', 'tutorial'))
                    or any(word in lower for word in ('deformable', 'improved', 'faster alternative'))):
                continue
            for match in re.finditer(r'https://github\.com/[\w.-]+/[\w.-]+', line):
                url = normalize_repository_url(match.group())
                if not url or url.casefold() in seen:
                    continue
                seen.add(url.casefold())
                candidates.append({'url': url, 'source': 'readme_linked_training',
                    'evidence_type': 'linked_implementation', 'is_author_code': False,
                    'context': '作者 README 明确链接的训练实现；需要单独检查方法与运行条件',
                    'relationship': {'path': path, 'quote': line.strip(), 'sha256': source['sha256']}})
    return candidates[:2]


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
    from src.author_dependencies import has_declaration
    from src.discovered_repository_plan import (CPU_INDEX_URL, CUDA_INDEX_URL, GRAPH_WHEEL_LINKS)
    from src.dependency_versions import compatibility_pins, reviewed_python_requirement
    device = (plan.get("capture") or {}).get("device", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("作者执行设备不在已审阅环境中")
    version_map = compatibility_pins(device)
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
                        has_declaration(value, requirements.get('author', [])))
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
            "cache_lock_timeout_s": AUTO_PREPARATION_LOCK_TIMEOUT,
            **reviewed_python_requirement(checked, device)}


_EXECUTION_PREPARATION = '''import json, platform, sys
from pathlib import Path
import torch, numpy
root = Path(__file__).resolve().parent
contract = json.loads((root / "config.json").read_text(encoding="utf-8"))["capture"]
record = {"python": sys.version, "platform": platform.platform(),
          "device": contract.get("device", "cpu"), "native_probe_pass": None,
          "status": "prepared", "checks": "disabled",
          "versions": {"torch": torch.__version__, "numpy": numpy.__version__}}
(root / "preflight.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record), flush=True)
'''


from src.agent_collaboration import collaborative_run, current_collaboration


class DiscoveredRepositoryReproduction:
    def __init__(self, root, logger, use_docker=False, llm=None):
        self.root, self.logger, self.use_docker, self.llm = Path(root), logger, use_docker, llm
        self.runner = RepositoryRunner(logger=logger)

    @collaborative_run
    @investigating_run
    @termination_signals()
    @release_run_environment
    def run(self, request, on_event=None):
        from src.run_lifecycle import cancellation_signals
        run_dir = self.root / "runs" / f"repository_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True)
        bind_environment_run(run_dir)
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
                if current_collaboration():
                    current_collaboration().persist(run_dir)

    def _run(self, request, on_event, run_dir):
        from src.agents.report_generator import ReportGeneratorAgent
        from src.discovered_repository_plan import PlanEvidenceError, build_evidence_packet, bind_full_paper_notes
        started = time.monotonic()
        emit = on_event or (lambda event: None)
        from src.strict_validation import normalize_strict_validation_request
        request = normalize_strict_validation_request(request)
        strict_requested = request['strict_validation']
        execution_only = not strict_requested
        phases = [("prepare_repository", "FIND_RESOURCES", "ResourceFinder", "获取代码仓库"),
                  ("plan_repository", "READ_PAPER", "PaperReader", "结合论文和源码规划作者命令"),
                  ("prepare_dataset", "FIND_RESOURCES", "ResourceFinder", "准备运行数据"),
                  ("prepare_environment", "BUILD_ENV", "EnvBuilder", "安装运行依赖"),
                  ("execute_repository", "EXECUTE_CODE", "CodeExecutor", "运行作者命令"),
                  ("finish_execution", "EXECUTE_CODE", "CodeExecutor", "记录命令退出与进程结束"),
                  ("compare_author_claims", "VALIDATE", "ResultValidator", "对比作者输出与论文指标"),
                  ("generate_report", "GENERATE_REPORT", "ReportGenerator", "生成运行报告")]
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
                "experiment_request": deepcopy(request['experiment_request']),
                "verifications": deepcopy(request.get("verifications") or []),
                "fix_records": deepcopy(request.get("fix_records") or []), "repository_executed": False,
                "reproduction_scope": "author_code_execution", "execution_source": "author_repository",
                "optimization": {"mode": "off", "available": False, "optimized": False,
                                 "status": "disabled", "reason": "运行已冻结作者实验协议"}}
        data['reproduction_scope'] = 'author_code_execution'
        collaboration = current_collaboration()
        if collaboration:
            collaboration.attach(data)
        investigation = current_investigation()
        if investigation:
            investigation.attach(data, run_dir)
        data['strict_validation'] = {'requested': strict_requested,
            'status': 'pending' if strict_requested else 'not_requested'}
        data['source_repair'] = {'requested': request['source_repair'],
            'auto': request.get('source_repair_auto', False),
            'editable': request['source_repair_editable'], 'max_rounds': 2,
            'status': 'pending' if request['source_repair'] else 'not_requested'}
        for key in ('reading_coverage', 'reading_batches', 'reading_originals'):
            if key in request:
                data[key] = deepcopy(request[key])
        phase, error = "prepare_repository", None
        before_calls = self.llm.get_call_count() if self.llm else 0
        try:
            event(phase, "running")
            if self.use_docker:
                raise ValueError("通用作者仓库执行当前需要本地受控进程模式")
            document, url, evidence = author_repository_input(
                {**request, 'author_execution_only': True}, require_author=False)
            manual = evidence["source"] == "user_supplied"
            if manual:
                data["execution_source"] = "user_supplied_repository"
                data["resources"].update(code_repo_url=url, selected_repo=url, selection_evidence=deepcopy(evidence),
                    repository_identity={"status": "user_supplied_unconfirmed", "source": "user_supplied_repository",
                        "author_identity_confirmed": False, "reason": "用户补充的仓库，论文与实现关系尚未核验", "evidence": deepcopy(evidence)})
            data["pdf_input"] = {"sha256": document.sha256, "pages": document.page_count,
                                 "bytes": document.byte_size, "readable": True,
                                 "extraction": document.extraction}
            if investigation:
                investigation.freeze_goal({'author_pdf_sha256': document.sha256,
                    'author_experiment_request': deepcopy(request['experiment_request'])})
            from src.pdf_markdown import write_markdown_evidence
            data['paper_markdown'] = write_markdown_evidence(run_dir, document)
            pin = (data["resources"].get("repo_discovery") or {}).get("pinned_revision", "")
            offline = bool(request.get("offline"))
            unique = repository_candidates(document, evidence, data['resources'], execution_only=execution_only)
            attempts, packet, selected, plan, last_plan_error = [], None, False, None, None
            workspace = None
            data['repository_identity_attempts'] = attempts
            data['planning_trace'] = []

            def record_candidate_identity(candidate_url, candidate_evidence, candidate_revision):
                data["resources"].update(code_repo_url=candidate_url, selected_repo=candidate_url,
                    selection_evidence=deepcopy(candidate_evidence),
                    repository_identity={"status": "unreviewed_execution", "source": "pinned_git_direct_execution",
                        "author_identity_confirmed": False,
                        "reason": "直接获取并运行所选仓库；不进行论文匹配审核。",
                        "evidence": deepcopy(candidate_evidence)})
                data["resources"].setdefault("repo_discovery", {}).update(
                    selected_repo=candidate_url, pinned_revision=candidate_revision,
                    repository_identity=data["resources"]["repository_identity"])

            def execute_candidate(candidate_url, candidate_evidence, candidate_plan, candidate_packet,
                                  candidate_snapshot, candidate_revision, candidate_pin_evidence):
                nonlocal execution_only
                execution_only = candidate_plan['scope'] == 'author_code_execution'
                if not execution_only:
                    from src.strict_validation import assert_strict_plan
                    data['strict_validation'].update(status='eligible',
                        binding=assert_strict_plan(candidate_plan, candidate_packet, workspace))
                    for key in ('reason', 'failure_kind', 'failure_phase', 'evidence', 'rejection_packet'):
                        data['strict_validation'].pop(key, None)
                data['reproduction_scope'] = candidate_plan['scope']
                record_candidate_identity(candidate_url, candidate_evidence, candidate_revision)
                data['resources']['implementation_selection'] = {
                    'initial_repository': url, 'selected_repository': candidate_url,
                    'selected_revision': candidate_revision,
                    'selection_reason': candidate_evidence.get('context') or candidate_evidence.get('source'),
                    'source_reference': deepcopy(candidate_evidence.get('relationship')),
                    'implementation_difference': ('选择了独立仓库实现；方法与训练入口需以该仓库固定源码为准'
                                                  if candidate_url.casefold() != url.casefold() else '原始候选仓库'),
                    'entrypoints': [deepcopy(step.get('entrypoint')) for step in candidate_plan['steps']]}
                data.update(repository=candidate_snapshot, revision_resolution=candidate_pin_evidence)
                write_json(run_dir / "repository.json", candidate_snapshot)
                event('plan_repository', "success")
                if digest(Path(request["pdf_path"])) != document.sha256:
                    changed = ValueError("原始 PDF 在仓库准备期间发生变化，不能使用不同论文继续规划")
                    changed.reproduction_phase = 'prepare_environment'
                    changed.category = 'source_changed'
                    raise changed
                write_json(run_dir / 'planning_trace.json', data['planning_trace'])
                data["grounded_repository_plan"] = candidate_plan
                data["spec_sha256"] = spec_digest(candidate_plan)
                write_json(run_dir / "experiment_spec.json", {"spec": candidate_plan, "sha256": data["spec_sha256"]})
                data["paper_info"]["metrics"] = {item["name"]: item["reference"] for item in candidate_plan["metrics"]}
                data["paper_info"]["required_metrics"] = [item["name"] for item in candidate_plan["metrics"]]
                candidate_environment = environment_for_plan(candidate_plan)
                from src.planning_recovery import validate_plan_consistency
                data['plan_consistency'] = validate_plan_consistency(
                    candidate_plan, candidate_packet, workspace, candidate_environment)
                write_json(run_dir / 'plan_consistency.json', data['plan_consistency'])
                if investigation:
                    investigation.remember_fact('validated_author_plan', {
                        'spec_sha256': data['spec_sha256'], 'pdf_sha256': candidate_plan['pdf_sha256'],
                        'repository': candidate_plan['repository'],
                        'experiment_request': request['experiment_request'],
                        'consistency': data['plan_consistency']})
                self._receive_author_plan('EnvBuilder', candidate_plan)
                candidate_environment["offline"] = offline
                data["env_config"] = deepcopy(candidate_environment)
                data["experiment_spec"] = {"id": "discovered_author_experiment", "adapter_id": "discovered",
                    "label": "依据本次 PDF 与用户提供的固定仓库生成的实验计划" if manual else "依据本次 PDF 与固定作者源码生成的实验计划",
                    "repository": candidate_plan["repository"],
                    "paper": deepcopy(data["paper_info"]), "parameters": candidate_plan.get("parameters", {}),
                    "environment": deepcopy(candidate_environment), "validation": {"scope": candidate_plan['scope'],
                        "note": ("核验原作者完整命令的实际运行和进程结束；尚未验收论文数值。" if execution_only else
                                 "冻结所选实验协议，捕获作者权重并在独立进程中复算完整测试集指标。")}}
                event('prepare_dataset', "running")
                datasets = prepare_plan_datasets(self.root, workspace, candidate_plan, candidate_snapshot, offline=offline)
                data["dataset_provenance"] = datasets
                write_json(run_dir / "dataset.json", datasets)
                event('prepare_dataset', 'success' if datasets['verified'] else 'waiting',
                      '' if datasets['verified'] else '等待固定依赖就绪后执行原始作者数据加载器')
                if execution_only and not candidate_plan['datasets']:
                    event('prepare_dataset', 'skipped', '输入数据由原作者程序加载；完整数据来源尚未核验，先验证代码可运行')
                self._execute_frozen(request, data, candidate_plan, candidate_snapshot, workspace,
                                     run_dir, candidate_environment, emit, event)

            index = 0
            while index < len(unique) and index < 3:
                candidate = unique[index]
                index += 1
                candidate_url = candidate["url"]
                attempt = {"repository_url": candidate_url, "accepted": False}
                attempts.append(attempt)
                data['planning_trace'].append(attempt)
                try:
                    def prepare_candidate():
                        nonlocal workspace, packet, plan, phase
                        self.logger.log("ResourceFinder", phase, "RUNNING", f"正在定位候选仓库版本（{index}/{len(unique)}）：{candidate_url}")
                        revision, pin_evidence = resolve_repository_revision(
                            candidate_url, pin if candidate_url.casefold() == url.casefold() else "", offline=offline)
                        workspace = run_dir / ("repo" if index == 1 else f"candidate_repos/{index}")
                        self.logger.log("ResourceFinder", phase, "RUNNING", "正在获取源码：" + revision)
                        snapshot = export_repository(self.root, {"repository": {"url": candidate_url, "revision": revision}},
                                                     workspace, offline=offline, required_files=[], include_submodules=False)
                        if snapshot.get("resolved_sha") != revision or snapshot.get("url") != candidate_url:
                            raise ValueError("导出的仓库与所选作者来源或固定版本不一致")
                        attempt.update(revision=revision, archive_sha256=snapshot.get("archive_sha256"),
                                       workspace=str(workspace.resolve()))
                        if not manual:
                            # Planning can fail before execute_candidate. Never let
                            # caller-supplied verification survive that error path.
                            record_candidate_identity(candidate_url, candidate, revision)
                        # Allow discovered/user-provided URLs without a literal
                        # author-code statement in the PDF. This flag only controls
                        # evidence acquisition; execution mode skips the reviewer.
                        self.logger.log("ResourceFinder", phase, "RUNNING", "源码已就绪；正在读取 README、依赖声明与论文运行线索")
                        packet = build_evidence_packet(request["pdf_path"], workspace, snapshot, for_identity_review=True)
                        packet['author_execution_only'] = not strict_requested
                        if request['experiment_request']:
                            packet['experiment_request'] = deepcopy(request['experiment_request'])
                        if not strict_requested:
                            packet['limits']['training_timeout_s'] = None
                        bind_full_paper_notes(packet, request)
                        if manual:
                            packet["repository_input"] = {"source": "user_supplied", "url": url}
                        elif candidate.get('source') == 'model_discovered':
                            packet['repository_input'] = {'source': 'model_discovered', 'url': candidate_url}
                        if packet["pdf"]["sha256"] != document.sha256:
                            raise ValueError("原始 PDF 在仓库准备期间发生变化，不能使用不同论文继续规划")
                        write_json(run_dir / f"repository_identity_packet_{index}.json", packet)
                        if snapshot.get('submodules_pending'):
                            materialize_submodules(self.root, snapshot, offline=offline)
                            packet = build_evidence_packet(request['pdf_path'], workspace, snapshot, for_identity_review=True)
                            packet['author_execution_only'] = not strict_requested
                            if request['experiment_request']:
                                packet['experiment_request'] = deepcopy(request['experiment_request'])
                            if not strict_requested:
                                packet['limits']['training_timeout_s'] = None
                            bind_full_paper_notes(packet, request)
                            packet['repository_input'] = {'source': candidate['source'], 'url': candidate_url}
                        packet['repository_identity_pending'] = False
                        event('prepare_repository', 'success')
                        phase = 'plan_repository'; event(phase, 'running')
                        data['resources']['selected_repo'] = candidate_url
                        write_json(run_dir / 'source_packet.json', packet)
                        if collaboration:
                            collaboration.send('ResourceFinder', 'AuthorPlanner', 'handoff', {
                                'repository': {'url': candidate_url, 'revision': revision},
                                'selection': candidate.get('source'),
                                'experiment_request': request['experiment_request'],
                                'pdf_sha256': packet['pdf']['sha256']})
                            collaboration.send('SourceReader', 'AuthorPlanner', 'evidence', {
                                'source_paths': list(packet.get('repository', {}).get('files') or {})[:40],
                                'inventory_count': len(packet.get('file_inventory') or {}),
                                'source_packet': 'source_packet.json',
                                'rule': 'Only hash-validated original source and paper evidence authorize a plan.'})
                        from src.author_execution_plan import propose_author_execution_plan
                        try:
                            if strict_requested:
                                from src.agents.experiment_protocol import ExperimentProtocolAgent
                                from src.discovered_repository_plan import propose_plan
                                author_packet = deepcopy(packet)
                                try:
                                    protocol = ExperimentProtocolAgent(self.llm, self.logger).run(packet, workspace)
                                    packet['experiment_protocol'] = protocol
                                    if collaboration:
                                        collaboration.send('ExperimentProtocol', 'AuthorPlanner', 'protocol', {
                                            'status': protocol.get('status'),
                                            'protocol': protocol,
                                            'authority': 'The frozen protocol is validated separately from messages.'})
                                    write_json(run_dir / 'experiment_protocol.json', protocol)
                                    plan = propose_plan(self.llm, packet, workspace)
                                    from src.strict_validation import assert_strict_plan
                                    assert_strict_plan(plan, packet, workspace)
                                except PlanEvidenceError as strict_error:
                                    # Eligibility limits the acceptance claim, not the
                                    # ability to run the original author command.
                                    # Never downgrade changed frozen inputs to a warning.
                                    verify_snapshot(workspace, snapshot, {})
                                    if digest(request['pdf_path']) != document.sha256:
                                        raise ValueError('原始 PDF 在严格规划期间发生变化') from strict_error
                                    rejection_packet = ('strict_rejection_packet.json' if index == 1
                                                        else f'strict_rejection_packet_{index}.json')
                                    write_json(run_dir / rejection_packet, packet)
                                    data['strict_validation'].update(status='unavailable', reason=str(strict_error),
                                        failure_kind=strict_error.category,
                                        rejection_packet=rejection_packet,
                                        evidence=deepcopy(getattr(strict_error, 'planning_evidence', None)))
                                    write_json(run_dir / 'strict_validation.json', data['strict_validation'])
                                    self.logger.log('PaperReader', 'strict_validation', 'WARNING',
                                        '严格验收暂不可用，继续运行原作者命令：' + str(strict_error))
                                    # Strict discovery must not spend the ordinary
                                    # planner's source budget or leave rejected
                                    # protocol state in its execution evidence.
                                    packet = author_packet
                                    packet['author_execution_only'] = True
                                    packet['limits']['training_timeout_s'] = None
                                    plan = propose_author_execution_plan(self.llm, packet, workspace)
                            else:
                                plan = propose_author_execution_plan(self.llm, packet, workspace)
                        except BaseException:
                            try:
                                write_json(run_dir / 'source_packet.json', packet)
                            except OSError as storage_error:
                                data['source_packet_persistence_error'] = str(storage_error)
                            raise
                        else:
                            write_json(run_dir / 'source_packet.json', packet)
                        return {'plan': plan, 'packet': packet, 'snapshot': snapshot,
                                'revision': revision, 'pin_evidence': pin_evidence}

                    from src.planning_recovery import alternative_candidate_attempt
                    prepared = alternative_candidate_attempt(prepare_candidate, candidate=candidate,
                        previous_attempts=attempts[:-1], pdf_sha256=document.sha256,
                        experiment_request=request['experiment_request'])
                    snapshot, revision, pin_evidence = (prepared[key] for key in
                        ('snapshot', 'revision', 'pin_evidence'))
                    execute_candidate(candidate_url, candidate, plan, packet, snapshot, revision, pin_evidence)
                    selected = True
                    attempt.update(accepted=True, revision=revision, planning_status='accepted',
                                   entrypoints=[deepcopy(step.get('entrypoint')) for step in plan['steps']],
                                   selection_reason=candidate.get('context') or candidate.get('source'))
                    url, evidence = candidate_url, deepcopy(candidate)
                    break
                except PlanEvidenceError as exc:
                    if getattr(exc, 'reproduction_phase', None):
                        attempt.update(error=str(exc), planning_status='execution_failed',
                                       failure_kind=getattr(exc, 'category', 'execution_failure'))
                        raise
                    selected, last_plan_error = False, exc
                    attempt.update(error=str(exc), planning_status='rejected',
                                   failure_kind=getattr(exc, 'category', 'invalid_evidence'))
                    if getattr(exc, 'planning_evidence', None) is not None:
                        attempt['planning_evidence'] = deepcopy(exc.planning_evidence)
                    if not manual and packet is not None:
                        linked = [item for item in linked_training_repositories(packet)
                                  if item['url'].casefold() not in {entry['url'].casefold() for entry in unique}]
                        unique[index:index] = linked
                    self.logger.log('PaperReader', phase, 'WARNING', '当前仓库规划受阻：' + str(exc))
                    event('plan_repository', 'skipped', str(exc)[:500])
                    phase = 'prepare_repository'; event(phase, 'running')
                except Exception as exc:
                    if isinstance(exc, InvestigationStopped):
                        raise
                    if getattr(exc, 'reproduction_phase', None) == 'prepare_environment':
                        attempt.update(error=str(exc), planning_status='environment_blocked',
                                       failure_kind=getattr(exc, 'category', 'environment_failure'))
                        if (getattr(exc, 'category', None) not in {'source_changed', 'installation_timeout'} and not manual
                                and packet is not None and not data.get('repository_executed')):
                            linked = [item for item in linked_training_repositories(packet)
                                      if item['url'].casefold() not in {entry['url'].casefold() for entry in unique}]
                            if linked and index < 3:
                                unique[index:index] = linked
                                self.logger.log('EnvBuilder', 'prepare_environment', 'WARNING',
                                                '当前实现环境受阻，检查 README 明确关联的训练实现：' + str(exc))
                                event('prepare_environment', 'skipped', str(exc)[:500])
                                phase = 'prepare_repository'
                                event(phase, 'running')
                                continue
                        raise
                    if getattr(exc, 'reproduction_phase', None):
                        attempt.update(error=str(exc), planning_status='execution_failed',
                                       failure_kind=getattr(exc, 'category', 'execution_failure'))
                        raise
                    if phase == 'plan_repository':
                        attempt.update(error=str(exc), planning_status='planner_error')
                        raise
                    selected = False
                    attempt.update(error=str(exc))
                    self.logger.log("ResourceFinder", phase, "WARNING", "当前候选准备失败：" + str(exc))
                    if digest(Path(request["pdf_path"])) != document.sha256:
                        raise ValueError("原始 PDF 在仓库准备期间发生变化，不能继续") from exc
                finally:
                    write_json(run_dir / "repository_identity_attempts.json", attempts)
                    write_json(run_dir / 'planning_trace.json', data['planning_trace'])
            data["repository_identity_attempts"] = attempts
            if not selected:
                if last_plan_error is not None:
                    phase = 'plan_repository'
                    evidence = getattr(last_plan_error, 'planning_evidence', {})
                    raise PlanEvidenceError(str(last_plan_error),
                        category=getattr(last_plan_error, 'category', 'missing_information'),
                        planning_evidence={**evidence, 'candidate_attempts': deepcopy(attempts)}) from last_plan_error
                raise ValueError("无法获取候选仓库源码。" + "；".join(attempt.get("error", "") for attempt in attempts))
        except (KeyboardInterrupt, SystemExit) as exc:
            phase = getattr(exc, "reproduction_phase", phase)
            if investigation:
                investigation.stop('cancelled', requires_human=False)
            data["interrupted"] = True
            if getattr(exc, "execution", None) is not None:
                data["environment_preparation" if phase == "prepare_environment" else "execution"] = exc.execution
            if phase == 'compare_author_claims' and data.get('validation', {}).get('status') == 'author_code_completed':
                # Cancellation stops optional analysis without undoing the
                # already persisted command/process completion evidence.
                comparison = data['validation'].setdefault('author_claim_comparison', {
                    'basis': 'author_output', 'metric_records': data['validation'].get('metric_records', []),
                    'comparisons': [], 'citations': []})
                comparison.update(status='interrupted', reason='作者代码已完成；论文指标对比已中断。')
                reason = comparison['reason']
            else:
                data["validation"] = {"status": "interrupted", "is_reproduced": None, "result_level": "failed",
                                      "reason": "实验已中断，未完成的训练和指标不能当作成功"}
                reason = data['validation']['reason']
            event(phase, "interrupted", reason)
            if collaboration:
                collaboration.attach(data)
                collaboration.persist(run_dir)
            write_json(run_dir / "result.json", {"state": "ERROR", "data": data, "error": "interrupted"})
            raise
        except Exception as exc:
            phase = getattr(exc, "reproduction_phase", phase)
            error = str(exc) or type(exc).__name__
            if investigation:
                # Retain concrete preparation blockers even when no controlled
                # repair is possible (missing hardware/data is not a new plan).
                investigation.record_issue(error, evidence={
                    'failure_kind': getattr(exc, 'category', 'runtime_error'),
                    'pdf_sha256': data.get('pdf_input', {}).get('sha256'),
                    'repository': data.get('resources', {}).get('selected_repo'),
                    'planning_evidence': getattr(exc, 'planning_evidence', None)},
                    role=next((item[2] for item in phases if item[0] == phase), 'Coordinator'))
            if phase == "prepare_repository":
                try:
                    fallback_pdf = extract_pdf_input(request["pdf_path"])
                    data["pdf_input"] = {"sha256": fallback_pdf.sha256, "pages": fallback_pdf.page_count,
                                         "bytes": fallback_pdf.byte_size, "readable": True}
                    data["repository_fallback"] = {"available": True, "pdf_sha256": fallback_pdf.sha256,
                        "reason": error, "action": "用户可补充 GitHub 仓库地址，直接获取并运行该仓库"}
                except (ValueError, OSError, KeyError):
                    pass
            if getattr(exc, "planning_evidence", None) is not None:
                data["plan_rejection"] = deepcopy(exc.planning_evidence)
                write_json(run_dir / "plan_rejection.json", data["plan_rejection"])
            data["validation"] = {"status": "execution_failed" if phase in {"execute_repository", "finish_execution"}
                                  else "insufficient_evidence", "failure_phase": phase, "is_reproduced": None,
                                  "failure_kind": getattr(exc, 'category', 'execution_failure' if phase in {'execute_repository','finish_execution'}
                                                          else 'runtime_error'),
                                  "result_level": "failed", "reason": error}
            data.setdefault("execution", {"mode": "repository", "executed": False, "success": False,
                                          "repository_executed": False})
            event(phase, "error", error)
        data["total_llm_calls"] = (self.llm.get_call_count() if self.llm else 0) - before_calls
        self.logger.add_llm_calls(data["total_llm_calls"])
        data["audit_stats"] = self.logger.get_stats()
        data["run_elapsed_s"] = time.monotonic() - started
        if collaboration:
            collaboration.send('CodeExecutor', 'ReportGenerator', 'result', {
                'execution': {key: (data.get('execution') or {}).get(key) for key in
                    ('success', 'executed', 'cancelled', 'timed_out', 'cleanup_confirmed')},
                'validation_status': (data.get('validation') or {}).get('status'),
                'error': error,
                'rule': 'Command completion does not establish independent numerical reproduction.'})
            collaboration.attach(data)
            collaboration.persist(run_dir)
        event("generate_report", "running")
        try:
            data.update(ReportGeneratorAgent(self.logger).run(data, report_path=data["report_path"]))
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

    @staticmethod
    def _receive_author_plan(role, plan):
        """Acknowledge the planner's evidence against the actual frozen input.

        The program receives a validated plan, not model-authored instructions
        from a mailbox. Stale or mismatched messages cannot replace that plan.
        """
        collaboration = current_collaboration()
        if collaboration is None:
            return
        from src.discovered_repository_plan import _canonical
        import hashlib
        expected = hashlib.sha256(_canonical(plan).encode('utf-8')).hexdigest()
        for message in collaboration.context(role):
            content = message.get('content')
            if (message['sender'] != 'AuthorPlanner' or message['kind'] != 'handoff'
                    or not isinstance(content, dict)):
                continue
            matched = content.get('plan_sha256') == expected
            collaboration.reply(message, role, {
                'status': 'accepted' if matched else 'ignored', 'plan_sha256': expected,
                'reason': ('交接与本次已校验的原始作者计划一致，按该计划处理。' if matched else
                           '交接不属于本次冻结计划，保留当前计划并忽略该消息。')}, kind='acknowledgement')

    def _execute_frozen(self, request, data, plan, snapshot, workspace, run_dir, environment, emit, event):
        """Materialize project capture utilities, then run and independently check."""
        self._receive_author_plan('CodeExecutor', plan)
        from src.discovered_repository_runtime import CAPTURE_DIR
        phase = "prepare_environment"
        try:
            event(phase, "running")
            execution_only = plan['scope'] == 'author_code_execution'
            from src.source_repair import validate_repair_sources
            validate_repair_sources(request, workspace, snapshot)
            with _frozen_input_validation():
                if read_json(run_dir / "experiment_spec.json") != {"spec": plan, "sha256": data["spec_sha256"]}:
                    raise ValueError("落盘实验协议与冻结的本次计划不一致")
                packet = read_json(run_dir / "source_packet.json")
                canonical = json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != plan["evidence_sha256"]:
                    raise ValueError("落盘来源包与冻结的实验依据不一致")
                if not execution_only:
                    from src.strict_validation import assert_strict_plan
                    assert_strict_plan(plan, packet, workspace)
            from src.author_python import select_author_python
            selected_python = select_author_python(packet, self.root / 'training_python',
                offline=bool(request.get('offline')),
                required_python=environment.get('required_python'),
                required_python_source=environment.get('required_python_source', 'actual_requested_dependency'))
            environment.update(selected_python)
            data.setdefault('training_python_attempts', []).append(deepcopy(selected_python))
            data['env_config'] = deepcopy(environment)
            write_json(run_dir / 'runtime_manifest.json', selected_python)
            if execution_only:
                from src.author_execution_plan import materialize_author_execution
                materialized = materialize_author_execution(workspace, plan, data['spec_sha256'], snapshot)
            else:
                from src.discovered_repository_runtime import materialize_capture
                materialized = materialize_capture(workspace, plan, data['spec_sha256'], snapshot=snapshot)
            generated = materialized["files"]
            runtime_dir = workspace / CAPTURE_DIR
            if not execution_only:
                configuration = read_json(runtime_dir / 'config.json')
                configuration['data_files'] = {
                    name: record for source in data['dataset_provenance']['sources']
                    for name, record in source.get('extracted_files', {}).items()}
                configuration['data_files'].update({source['target']: {
                    'sha256': source['sha256'], 'bytes': source['bytes']}
                    for source in data['dataset_provenance']['sources'] if source['kind'] == 'https'})
                write_json(runtime_dir / 'config.json', configuration)
                generated[f'{CAPTURE_DIR}/config.json'] = digest(runtime_dir / 'config.json')
            (runtime_dir / "preflight.py").write_text(
                _EXECUTION_PREPARATION if execution_only else
                (Path(__file__).parent / 'experiments/strict_preflight.py').read_text(encoding='utf-8'),
                encoding="utf-8")
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
            with _frozen_input_validation():
                verify_snapshot(workspace, snapshot, generated)
            steps = materialized["steps"]
            execution_plan = build_plan(profile="discovered_author_experiment", spec_sha256=data["spec_sha256"],
                workspace=workspace, repository=plan["repository"], dataset=data["dataset_provenance"], steps=steps)
            data["execution_plan"] = execution_plan
            write_json(run_dir / "execution_plan.json", execution_plan)
            write_json(run_dir / "environment.json", environment)
            def verify_frozen_inputs():
                with _frozen_input_validation():
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
                    if not execution_only:
                        assert_strict_plan(plan, stored_packet, workspace)
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
                if request.get('source_repair'):
                    data['source_repair'].update(status='not_run', reason='仅准备计划，尚未运行或请求源码补丁。')
                data["execution"] = {"mode": "repository", "executed": False, "success": False,
                                     "repository_executed": False, "reason": "仅准备计划，尚未执行作者训练"}
                data["validation"] = {"status": "prepared", "is_reproduced": None,
                    "result_level": "prepared", "reason": ('源码与协议已冻结；作者加载器数据仍待依赖准备后下载，尚未执行训练'
                    if loaders else '源码和完整原命令已冻结；实际数据将在作者执行中构造，尚未训练'
                    if execution_only else '源码和完整原命令已冻结；实际数据将在作者执行中构造，尚未训练'
                    if runtime_loaders else '源码、完整真实数据与协议已冻结，尚未执行训练')}
                event(phase, "success", "执行计划已准备，未安装或训练")
                if loaders or runtime_loaders:
                    event('prepare_dataset', 'skipped', '仅准备计划；作者加载器尚未下载真实数据')
                for identifier in ("execute_repository", "finish_execution", "compare_author_claims"):
                    event(identifier, "skipped", "仅准备模式")
                return

            def progress(item):
                emit({**item, "phase_id": phase})

            def repair_numeric(result):
                # Only an actual failed author observer can propose this one
                # deterministic repair. Ordinary exceptions remain failures.
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
                            or candidate.get('step') != identifier
                            or type(candidate.get('optimizer_steps')) is not int or candidate['optimizer_steps'] < 0
                            or fix.get('kind') != 'numpy_ragged_summary' and candidate['optimizer_steps'] != 0
                            or fix.get('kind') not in ('numpy_float_reciprocal', 'numpy_builtin_alias', 'numpy_ragged_summary')
                            or fix.get('source_sha256') != snapshot['files'].get(fix.get('path'))):
                        return None
                    from src.experiments.author_numeric_compatibility import BUILTIN_ALIASES, power_call, validate_ragged_repair
                    import ast
                    source = workspace_path(workspace, fix['path'], 'numeric compatibility source', must_exist=True)
                    tree = ast.parse(source.read_bytes())
                    if fix['kind'] == 'numpy_float_reciprocal':
                        call = power_call(tree, fix.get('line'), fix.get('column'))
                        if (call is None or ast.get_source_segment(source.read_text('utf-8'), call) != fix.get('original_expression')
                                or fix.get('replacement') != call.func.value.id + '.float_power'):
                            return None
                    elif fix['kind'] == 'numpy_builtin_alias':
                        alias = fix.get('alias')
                        matches = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                                   and node.attr == alias and node.lineno == fix.get('line')
                                   and node.col_offset == fix.get('column')
                                   and ast.get_source_segment(source.read_text('utf-8'), node) == fix.get('original_expression')]
                        if alias not in BUILTIN_ALIASES or len(matches) != 1 or fix.get('replacement') != 'builtins.' + alias:
                            return None
                    else:
                        try:
                            validate_ragged_repair(tree, fix, source.read_text('utf-8'))
                        except ValueError:
                            return None
                    capture_config = read_json(runtime_dir / 'config.json')
                    configured = capture_config.setdefault('numeric_compatibility', [])
                    if any(item['path'] == fix['path'] and item.get('line') == fix.get('line')
                           and item.get('column') == fix.get('column') and item['kind'] == fix['kind'] for item in configured):
                        return None
                    verify_frozen_inputs()
                    configured.append(fix)
                    write_json(runtime_dir / 'config.json', capture_config)
                    generated[f'{CAPTURE_DIR}/config.json'] = digest(runtime_dir / 'config.json')
                    write_json(run_dir / 'adapter_manifest.json', {'files': generated,
                        'note': '原仓库文件不变；记录实际失败后的 NumPy 兼容修复及原始表达式。'})
                    return fix
                return None

            def repair_path(result):
                from src.experiments.author_path_compatibility import validate_root_repair
                import ast
                for failed in result.get('steps', []):
                    if failed.get('executed') is not True or failed.get('exit_code') in (None, 0):
                        continue
                    identifier = failed.get('id')
                    if not any(step['id'] == identifier for step in plan['steps']):
                        continue
                    candidate_path = runtime_dir / ('path_failure_' + identifier + '.json')
                    if not candidate_path.exists():
                        continue
                    candidate = read_json(candidate_path)
                    fix = candidate.get('repair') or {}
                    if (candidate.get('spec_sha256') != data['spec_sha256']
                            or candidate.get('step') != identifier or candidate.get('optimizer_steps') != 0
                            or fix.get('source_sha256') != snapshot['files'].get(fix.get('path'))):
                        return None
                    source = workspace_path(workspace, fix['path'], 'author data root source', must_exist=True)
                    try:
                        validate_root_repair(ast.parse(source.read_bytes()), fix, source, workspace, snapshot['files'])
                    except ValueError:
                        return None
                    capture_config = read_json(runtime_dir / 'config.json')
                    configured = capture_config.setdefault('path_compatibility', [])
                    if any(item['path'] == fix['path'] and item['line'] == fix['line']
                           and item['column'] == fix['column'] for item in configured):
                        return None
                    verify_frozen_inputs()
                    configured.append(fix)
                    write_json(runtime_dir / 'config.json', capture_config)
                    generated[f'{CAPTURE_DIR}/config.json'] = digest(runtime_dir / 'config.json')
                    write_json(run_dir / 'adapter_manifest.json', {'files': generated,
                        'note': '原仓库文件不变；记录实际失败后的任务内数据目录兼容修复。'})
                    return fix
                return None

            def repair_pyg(result):
                from src.experiments.author_pyg_compatibility import validate_repair
                for failed in result.get('steps', []):
                    if failed.get('executed') is not True or failed.get('exit_code') in (None, 0):
                        continue
                    identifier = failed.get('id')
                    if not any(step['id'] == identifier for step in plan['steps']):
                        continue
                    candidate_path = runtime_dir / ('pyg_failure_' + identifier + '.json')
                    if not candidate_path.exists():
                        continue
                    candidate = read_json(candidate_path)
                    fix = candidate.get('repair') or {}
                    if (candidate.get('spec_sha256') != data['spec_sha256'] or candidate.get('step') != identifier
                            or candidate.get('optimizer_steps') != 0):
                        return None
                    try:
                        validate_repair(workspace, snapshot['files'], fix)
                    except (ValueError, OSError, TypeError):
                        return None
                    capture_config = read_json(runtime_dir / 'config.json')
                    configured = capture_config.setdefault('pyg_compatibility', [])
                    if fix in configured:
                        return None
                    verify_frozen_inputs()
                    configured.append(fix)
                    write_json(runtime_dir / 'config.json', capture_config)
                    generated[f'{CAPTURE_DIR}/config.json'] = digest(runtime_dir / 'config.json')
                    write_json(run_dir / 'adapter_manifest.json', {'files': generated,
                        'note': '原作者源码与数据未改写；记录实际失败后的旧 PyG 内存容器兼容。'})
                    return fix
                return None

            def prepare_runtime_environment(effective):
                from packaging.specifiers import SpecifierSet
                from src.dependency_versions import reviewed_python_requirement
                required = reviewed_python_requirement(effective.get('requirements_txt', '').splitlines(),
                    plan['capture'].get('device', 'cpu'))
                if not required:
                    return effective
                prior = ([{'path': effective.get('required_python_source', 'actual_requested_dependency'),
                           'specifier': effective['required_python']}]
                         if effective.get('required_python') and effective['required_python'] != required['required_python']
                         else [])
                effective.update(required)
                # This reviewed constraint describes a whole minor family.
                # Full patch-level author declarations were checked at selection.
                if SpecifierSet(required['required_python']).contains(effective.get('python_version', '0')):
                    return effective
                replacement = select_author_python(packet, self.root / 'training_python',
                    offline=bool(request.get('offline')), additional_python_requirements=prior, **required)
                effective.update(replacement, auto_prepare=True, require_prepared=False)
                data['training_python_attempts'].append(deepcopy(replacement))
                data['env_config'] = deepcopy(effective)
                write_json(run_dir / 'runtime_manifest.json', replacement)
                write_json(run_dir / 'environment.json', effective)
                progress({'type': 'log', 'log': {'agent': 'CodeExecutor',
                    'action': 'select_native_training_python', 'status': 'RUNNING',
                    'detail': '实际新增依赖需要已验证的原生 CPU Python 组合；释放旧环境后重试原作者命令。',
                    'data': replacement}})
                return effective

            def run_steps(selected, runtime_environment):
                if current_investigation():
                    current_investigation().ensure_running()
                from src.author_runtime_dependencies import run_with_dependency_repair
                from src.author_failure_analysis import analyze_author_failure
                def analyze_failure(result, effective):
                    investigation = current_investigation()
                    if investigation:
                        task = investigation.investigate(trigger='execution_failure', plan=plan, packet=packet,
                            workspace=workspace, environment=effective, execution=result)
                        investigation.ensure_running()
                        if task and task['status'] != 'repair_proposed':
                            return None
                    before = len(data.get('failure_analyses') or [])
                    suggestion = analyze_author_failure(investigation.wrap_llm() if investigation else self.llm,
                                                        result, plan, packet, workspace,
                                                        effective, run_dir, data)
                    if len(data.get('failure_analyses') or []) > before:
                        analysis = data['failure_analyses'][-1]
                        progress({'type': 'log', 'log': {'agent': 'CodeExecutor',
                            'action': 'analyze_author_failure', 'status': 'SUCCESS',
                            'detail': analysis.get('reason') or analysis.get('analysis_error')
                                      or '已分析当前失败并记录所需条件',
                            'data': {'failure_kind': analysis['failure_kind'],
                                     'action': analysis['action'], 'llm_calls': analysis['llm_calls']}}})
                    return suggestion
                investigation = current_investigation()
                identifiers = {step.get('id') for step in selected} if isinstance(selected, list) else set()
                recovery_phase = ('environment_preparation' if 'native_preflight' in identifiers else
                                  'dataset_preparation' if 'prepare_author_datasets' in identifiers else
                                  'author_execution')
                with investigation.phase(recovery_phase) if investigation else nullcontext():
                    result = run_with_dependency_repair(self.runner, workspace, selected, runtime_environment,
                        plan, packet, run_dir, data, progress, verify_frozen_inputs, repair_numeric, repair_path,
                        analyze_failure, repair_pyg=repair_pyg, prepare_environment=prepare_runtime_environment)
                    if investigation:
                        investigation.record_execution(result)
                        investigation.ensure_running()
                    return result

            preflight = [{"id": "native_preflight", "kind": "check", "required": True,
                          "argv": ["python", "-u", f"{CAPTURE_DIR}/preflight.py"], "cwd": ".",
                          "timeout_s": 180, "env": {"CUDA_VISIBLE_DEVICES": ""}
                              if plan["capture"].get("device", "cpu") == "cpu" else {},
                          "artifacts": [f"{CAPTURE_DIR}/preflight.json"]}]
            preparation = run_steps(preflight, environment)
            from src.author_python import retry_requested_python_conflict
            def record_python_switch(replacement):
                data['training_python_attempts'].append(deepcopy(replacement))
                data['env_config'] = deepcopy(environment)
                write_json(run_dir / 'runtime_manifest.json', replacement)
            preparation = retry_requested_python_conflict(packet, self.root / 'training_python',
                environment, selected_python, preparation,
                lambda selected_environment: run_steps(preflight, selected_environment),
                offline=bool(request.get('offline')), on_switch=record_python_switch)
            data["environment_preparation"] = preparation
            write_json(run_dir / "environment_preparation.json", preparation)
            if preparation.get("success") is not True:
                raise RuntimeError("依赖准备或原生导入/运算检查失败：" + str(
                    preparation.get("reason") or preparation.get("final", {}).get("stderr", ""))[-2000:])
            environment = preparation.get('effective_env_config') or environment
            probe = read_json(runtime_dir / "preflight.json")
            data["native_preflight"] = probe
            if not execution_only and (probe.get('native_probe_pass') is not True
                                       or probe.get('device') != plan['capture'].get('device', 'cpu')):
                raise ValueError('严格运行环境缺少冻结设备上的实际原生运算证据')
            verify_frozen_inputs()
            event(phase, "success")
            if loaders:
                phase = 'prepare_dataset'; event(phase, 'running')
                dataset_step = [{'id': 'prepare_author_datasets', 'kind': 'check', 'required': True,
                    'argv': ['python', '-u', f'{CAPTURE_DIR}/prepare_datasets.py'], 'cwd': '.', 'timeout_s': None,
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
                capture_configuration.setdefault('data_files', {}).update(
                    {name: record for source in records for name, record in source['files'].items()})
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
                if request.get('source_repair'):
                    data['source_repair'].update(status='not_run', reason='仅准备环境，尚未运行或请求源码补丁。')
                data["execution"] = {"mode": "repository", "executed": False, "success": False,
                                     "repository_executed": False, "reason": "环境已准备，尚未执行作者训练"}
                data["validation"] = {"status": "environment_ready", "is_reproduced": None,
                                      "result_level": "prepared", "reason": ('原生环境已准备；原作者命令尚未构造实际数据或开始训练'
                                      if native_pending or execution_only else '真实数据、协议和原生环境已准备，尚未执行训练')}
                if runtime_loaders:
                    event('prepare_dataset', 'skipped', '仅准备环境；实际数据构造和训练尚未执行')
                for identifier in ("execute_repository", "finish_execution", "compare_author_claims"):
                    event(identifier, "skipped", "仅准备实验环境")
                return
            phase = "execute_repository"; event(phase, "running")
            verify_frozen_inputs()
            # Running author code has no elapsed-time budget. Only cancellation,
            # its exit status and the bounded error-repair loop can end execution.
            prepared_environment = {**environment, "auto_prepare": False, "require_prepared": True,
                                    "deadline_monotonic": None}
            execution = run_steps(execution_plan, prepared_environment)
            data['execution'] = execution
            if request.get('source_repair'):
                from src.source_repair import repair_author_failure
                data['original_execution'] = deepcopy(execution)
                write_json(run_dir / 'original_execution.json', execution)
                execution = repair_author_failure(self.runner, self.llm, request, plan, snapshot,
                    workspace, run_dir, data, execution, verify_frozen_inputs, emit)
            if current_investigation():
                current_investigation().record_execution(execution)
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
            phase = "finish_execution"; event(phase, "running")
            verify_frozen_inputs()
            if not execution_only:
                from src.strict_validation import verify_strict_execution
                validation = verify_strict_execution(plan, packet, workspace, execution, data['spec_sha256'])
                validation['dataset_provenance_verified'] = data['dataset_provenance']['verified']
                data['strict_validation'].update(status='verified', result=validation['status'])
                data['validation'] = validation
                execution['strict_execution_verification'] = validation
                write_json(run_dir / 'strict_validation.json', data['strict_validation'])
                write_json(run_dir / 'strict_execution_verification.json', validation)
                write_json(run_dir / 'execution.json', execution)
                event(phase, 'success', validation['reason'])
                event('compare_author_claims', 'success', '已使用独立复算结果对比冻结的论文参考值')
                _diagnose_outcome(self.llm, plan, packet, workspace, execution,
                                  validation, run_dir, data, emit)
                return
            from src.author_execution_plan import verify_author_execution
            repaired = data['source_repair']['status'] == 'succeeded'
            if repaired:
                from src.source_repair import verify_repaired_execution
                validation = verify_repaired_execution(plan, data, execution)
            else:
                validation = verify_author_execution(plan, workspace, execution, data['spec_sha256'])
            validation.update(reviews_removed=True, paper_metrics_verified=False,
                              pre_execution_review='disabled', dataset_provenance_verified=False)
            if data.get('strict_validation', {}).get('requested'):
                validation['strict_validation'] = deepcopy(data['strict_validation'])
                validation.setdefault('warnings', []).append('作者命令可运行；严格验收未完成：' +
                    data['strict_validation'].get('reason', '缺少完整独立证据'))
            if runtime_loaders and not repaired:
                from src.runtime_dataset_evidence import audit_runtime_datasets
                dataset_audit = audit_runtime_datasets(plan, validation['author_execution'], workspace)
                data['runtime_dataset_evidence'] = validation['runtime_dataset_evidence'] = dataset_audit
                validation['dataset_observation_verified'] = dataset_audit['verified']
                data['dataset_provenance']['runtime_observation'] = dataset_audit
                data['dataset_provenance']['reason'] = ('实际输入已按原加载器观测并在运行结束后核对；尚未与独立官方参考摘要核对。'
                    if dataset_audit['verified'] else '原作者代码已结束，完整输入观测未通过或尚未覆盖；保留未知状态。')
                for filename, value in [('runtime_dataset_evidence.json', dataset_audit),
                                        ('dataset.json', data['dataset_provenance'])]:
                    try:
                        write_json(run_dir / filename, value)
                    except OSError as exc:
                        dataset_audit.setdefault('persistence_errors', []).append({'file': filename, 'reason': str(exc)})
                        validation.setdefault('warnings', []).append('输入观测记录未能单独保存：' + filename + ': ' + str(exc))
                if dataset_audit['errors']:
                    validation.setdefault('warnings', []).append('原作者输入观测未完成核对：' +
                        '; '.join(item['reason'] for item in dataset_audit['errors']))
                event('prepare_dataset', 'success' if dataset_audit['verified'] else 'skipped',
                      data['dataset_provenance']['reason'])
            if validation.get('numeric_compatibility'):
                kinds = {item['kind'] for item in validation['numeric_compatibility']}
                notes = []
                if 'numpy_float_reciprocal' in kinds:
                    notes.append('浮点倒数兼容修复')
                if 'numpy_builtin_alias' in kinds:
                    notes.append('旧 NumPy 类型别名兼容修复')
                if 'numpy_ragged_summary' in kinds:
                    notes.append('旧 NumPy 不规则结果表汇总兼容修复')
                validation.setdefault('warnings', []).append('执行包含实际报错后记录的' + '、'.join(notes) + '；原仓库文件未改写。')
            if validation.get('path_compatibility'):
                validation.setdefault('warnings', []).append('作者缺失的临时数据目录已在实际报错后映射到本任务内；原仓库文件未改写。')
            if validation.get('pyg_compatibility'):
                validation.setdefault('warnings', []).append('执行包含实际报错后的旧 PyG 容器兼容；原数据文件、张量值与划分未改写。')
            from src.author_claim_comparison import collect_author_metrics
            metric_workspace = Path(data['source_repair']['workspace']) if repaired else workspace
            metric_sources = data['source_repair']['effective_source_files'] if repaired else snapshot['files']
            try:
                validation['metric_records'], data['metric_parse_errors'] = collect_author_metrics(
                    execution, workspace=metric_workspace, plan=plan, sources=metric_sources)
                if any(record.get('uncertainty_kind') == 'bootstrap_confidence_interval' for record in validation['metric_records']):
                    validation.setdefault('warnings', []).append('作者打印的误差项为 bootstrap 置信区间半宽，未当作标准差比较。')
            except Exception as exc:
                validation['metric_records'] = []
                data['metric_parse_errors'] = [str(exc)]
            data['validation'] = validation
            execution['author_execution_verification'] = validation
            write_json(run_dir / 'author_execution_verification.json', validation)
            write_json(run_dir / 'execution.json', execution)
            event(phase, 'success', validation['reason'])
            # Author completion is established and saved before asking the
            # model for an informational, citation-checked paper comparison.
            phase = 'compare_author_claims'
            from src.author_claim_comparison import compare_author_claims, TOLERANCE
            comparison = {'status': 'unavailable', 'basis': 'author_output', 'relative_tolerance': TOLERANCE,
                          'metric_records': validation['metric_records'], 'parse_errors': data['metric_parse_errors'],
                          'comparisons': [], 'citations': [],
                          'reason': '论文指标对比暂未完成；保留实际指标，代码运行已完成。'}
            validation['author_claim_comparison'] = comparison
            event(phase, 'running')
            try:
                comparison.update(compare_author_claims(self.llm, packet, plan, workspace, execution,
                                                        sources=metric_sources, metric_workspace=metric_workspace))
            except Exception as exc:
                comparison['diagnostic'] = str(exc) or type(exc).__name__
            if repaired:
                comparison.update(basis='patched_author_output', source_modified=True,
                    source_repair_manifest_sha256=data['source_repair']['manifest_sha256'])
                comparison['reason'] += ' 本次实际输出来自兼容补丁后的源码，未作为原始源码或独立论文复现证明。'
            for name, value in (('author_claim_comparison.json', comparison),
                                ('author_execution_verification.json', validation), ('execution.json', execution)):
                try:
                    write_json(run_dir / name, value)
                except OSError as exc:
                    comparison['persistence_error'] = str(exc)
                    validation.setdefault('warnings', []).append(f'论文指标对比记录保存失败（{name}）：{exc}')
            event(phase, 'success' if comparison['comparisons'] else 'skipped', comparison['reason'])
            _diagnose_outcome(self.llm, plan, packet, workspace, execution,
                              validation, run_dir, data, emit)
        except BaseException as exc:
            if data.get('strict_validation', {}).get('status') == 'eligible':
                data['strict_validation'].update(status='failed', failure_phase=phase,
                                               reason=str(exc) or type(exc).__name__)
            if phase == 'prepare_environment':
                effective = (data.get('environment_preparation') or {}).get('effective_env_config') or {}
                if effective.get('dependency_failure_kind') and getattr(exc, 'category', None) != 'source_changed':
                    exc.category = effective['dependency_failure_kind']
            exc.reproduction_phase = phase
            raise

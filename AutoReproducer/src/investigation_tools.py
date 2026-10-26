"""Fixed observation tools. No model-supplied executable or expected value."""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import tempfile

from src.safety.paths import workspace_path


MAX_SOURCE_BYTES = 2 * 1024 * 1024
TOOLS = ('source_context', 'runtime_metadata', 'artifact_contract', 'author_contract')


class InvestigationTools:
    def __init__(self, workspace, packet, environment, run_dir, runner=None):
        self.workspace = Path(workspace)
        self.packet, self.environment = packet, environment
        self.run_dir = Path(run_dir) if run_dir else None
        self._candidate_sources = {}

    @contextmanager
    def candidate_contracts(self, proposal):
        """Controller-only patch context; model tool arguments cannot replace tests."""
        previous = self._candidate_sources
        candidates = {}
        for patch in proposal.get('patches') or []:
            name = patch.get('path')
            if not self.readable_source(name, (self.packet.get('file_inventory') or {}).get(name)):
                continue
            _, sha = self._source(name)
            if patch.get('base_sha256') != sha or not isinstance(patch.get('content'), str):
                raise ValueError('Candidate contract source does not match the frozen original')
            candidates[name] = patch['content']
        self._candidate_sources = candidates
        try:
            yield
        finally:
            self._candidate_sources = previous

    def execute(self, tool, arguments):
        if tool not in TOOLS or not isinstance(arguments, dict):
            return self._result(tool, 'rejected', [], 'Unknown fixed tool or invalid arguments.')
        try:
            return getattr(self, '_' + tool)(arguments)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            from src.runtime_preparation import RuntimePreparationTimeout
            cleanup_confirmed = getattr(exc, 'process_cleanup_confirmed', True) is not False
            stopped = not cleanup_confirmed or isinstance(exc, RuntimePreparationTimeout)
            result = self._result(tool, 'stopped' if stopped else 'rejected', [],
                                 'Fixed observation unavailable: ' + type(exc).__name__, cleanup_confirmed)
            if isinstance(exc, RuntimePreparationTimeout):
                result['timed_out'] = True
            return result

    @staticmethod
    def _result(tool, status, evidence, summary, cleanup_confirmed=True):
        return {'tool': tool, 'status': status, 'evidence': evidence,
                'summary': summary, 'cleanup_confirmed': cleanup_confirmed,
                'authority': 'Observations only; no execution success or numerical equivalence claim.'}

    def _source(self, name):
        inventory = self.packet.get('file_inventory') or {}
        if not isinstance(name, str) or name not in inventory:
            raise ValueError('Source is not in frozen inventory')
        path = workspace_path(self.workspace, name, 'investigation source', must_exist=True, forbid_git=True)
        if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError('Source exceeds observation budget')
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if inventory[name].get('sha256') != sha:
            raise ValueError('Frozen source changed')
        return content, sha

    @staticmethod
    def _paths(arguments):
        paths = arguments.get('paths')
        if (set(arguments) != {'paths'} or not isinstance(paths, list) or not 1 <= len(paths) <= 5
                or any(not isinstance(name, str) for name in paths)):
            raise ValueError('Expected one to five frozen paths')
        return paths

    @staticmethod
    def readable_source(name, record):
        if not isinstance(name, str) or not isinstance(record, dict):
            return False
        path = Path(name)
        if any(part.lower() in {'private', 'hidden', 'hidden_labels'} for part in path.parts):
            return False
        if record.get('text_available') is not True:
            return False
        if path.suffix.lower() == '.py':
            return True
        if any(part.lower() in {'data', 'datasets', 'labels', 'targets', 'predictions'} for part in path.parts[:-1]):
            return False
        if path.suffix.lower() == '.txt':
            return path.name.lower().startswith(('readme', 'requirements', 'constraints'))
        return path.suffix.lower() in {'.md', '.rst', '.toml', '.cfg', '.ini', '.yaml', '.yml', '.ipynb'}

    def _source_context(self, arguments):
        if set(arguments) == {'paper_pages'}:
            requested = arguments['paper_pages']
            if (not isinstance(requested, list) or not 1 <= len(requested) <= 5
                    or any(type(page) is not int or page < 1 for page in requested)):
                raise ValueError('Expected one to five original PDF page numbers')
            pdf = self.packet.get('pdf') or {}
            if not isinstance(pdf.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', pdf['sha256']):
                raise ValueError('Frozen paper identity is unavailable')
            pages = {page['page']: page for page in pdf.get('pages') or []}
            evidence = []
            for number in dict.fromkeys(requested):
                page = pages[number]
                sha = hashlib.sha256(page['text'].encode('utf-8')).hexdigest()
                if sha != page.get('sha256'):
                    raise ValueError('Frozen PDF page text changed')
                evidence.append({'id': 'paper_' + sha[:16] + '_p' + str(number),
                    'source': 'frozen_pdf_page', 'sha256': sha, 'pdf_sha256': pdf['sha256'],
                    'value': {'page': number, 'text': page['text'][:16000], 'truncated': len(page['text']) > 16000}})
            return self._result('source_context', 'completed', evidence, 'Read selected hash-matching original PDF context.')
        evidence = []
        for name in self._paths(arguments):
            if not self.readable_source(name, (self.packet.get('file_inventory') or {}).get(name)):
                raise ValueError('Only public frozen source/configuration text is readable; data requires metadata tools')
            content, sha = self._source(name)
            text = content.decode('utf-8', 'replace')
            if name.endswith('.ipynb'):
                from src.author_entrypoints import notebook_cells
                text = '\n'.join(cell['source'] for cell in notebook_cells(self.packet, self.workspace, name))
            value = {'path': name, 'text': text[:16000], 'truncated': len(text) > 16000}
            if name.endswith('.py'):
                tree = ast.parse(text)
                value['imports'] = [ast.unparse(node) for node in ast.walk(tree)
                                    if isinstance(node, (ast.Import, ast.ImportFrom))][:100]
                value['definitions'] = [{'name': node.name, 'line': node.lineno,
                                         'arguments': ast.unparse(node.args) if hasattr(node, 'args') else None}
                                        for node in ast.walk(tree) if isinstance(node,
                                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))][:100]
            evidence.append({'id': 'source_' + sha[:16] + '_' + hashlib.sha256(name.encode()).hexdigest()[:8],
                             'source': name, 'sha256': sha, 'value': value})
        return self._result('source_context', 'completed', evidence, 'Read hash-matching source and static definitions.')

    def _child(self, payload):
        from src.runtime_preparation import run_owned_process
        python = self.environment.get('python_executable')
        if not isinstance(python, str) or not Path(python).is_absolute() or not Path(python).is_file():
            raise ValueError('A verified author Python is required; application Python is not a fallback')
        directory = self.run_dir / 'investigation_probes' if self.run_dir else None
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        helper = Path(__file__).parent / 'experiments' / 'investigation_probe.py'
        cleaned = []
        with tempfile.TemporaryDirectory(prefix='probe_', dir=directory) as isolated:
            result = run_owned_process([python, '-I', '-S', str(helper.resolve())], cwd=isolated,
                input=json.dumps(payload), timeout_s=60, max_output_bytes=65536,
                on_cleanup=lambda: cleaned.append(True))
        if not cleaned:
            failure = RuntimeError('Probe did not finish with a confirmed process cleanup')
            failure.process_cleanup_confirmed = False
            failure.cleanup_receipt = {'status': 'unconfirmed', 'process_cleanup_confirmed': False}
            raise failure
        if result.returncode != 0:
            raise ValueError('Fixed probe helper failed')
        response = json.loads(result.stdout)
        if not isinstance(response, dict):
            raise ValueError('Invalid fixed helper output')
        return response

    def _runtime_metadata(self, arguments):
        if set(arguments) - {'packages', 'modules'}:
            raise ValueError('Unknown metadata arguments')
        packages, modules = arguments.get('packages', []), arguments.get('modules', [])
        if (not isinstance(packages, list) or not isinstance(modules, list)
                or not 1 <= len(packages) + len(modules) <= 8
                or any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', value) for value in packages)
                or any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', value) for value in modules)):
            raise ValueError('Expected bounded package/module identifiers')
        result = self._child({'tool': 'runtime_metadata', 'packages': packages, 'modules': modules,
                             'metadata_paths': self.environment.get('investigation_dependency_paths') or []})
        # Local module origin is discovered from inventory, never by importing it.
        inventory = self.packet.get('file_inventory') or {}
        result['local_modules'] = {}
        for module in modules:
            stem = module.replace('.', '/')
            candidates = [stem + '.py', stem + '/__init__.py']
            found = []
            for name in candidates:
                if name in inventory:
                    _, sha = self._source(name)
                    found.append({'path': name, 'sha256': sha})
            result['local_modules'][module] = found
        sha = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return self._result('runtime_metadata', 'completed', [
            {'id': 'runtime_' + sha[:16], 'source': 'isolated_author_python_metadata', 'sha256': sha, 'value': result}],
            'Observed author Python distributions and static module origins; no author imports executed.')

    def _artifact_contract(self, arguments):
        evidence = []
        artifacts = self.packet.get('investigation_artifacts') or {}
        for name in self._paths(arguments):
            if any(part.lower() in {'hidden', 'private', 'hidden_labels'} for part in Path(name).parts):
                raise ValueError('Private evaluator inputs are not investigator evidence')
            record = artifacts.get(name)
            if not isinstance(record, dict) or record.get('public') is not True:
                raise ValueError('Artifact is not in the explicit public input/output manifest')
            if name in (self.packet.get('file_inventory') or {}):
                content, sha = self._source(name)
                if sha != record.get('sha256'):
                    raise ValueError('Public artifact identity changed')
            else:
                path = workspace_path(self.workspace, name, 'investigation artifact', must_exist=True, forbid_git=True)
                if path.stat().st_size > MAX_SOURCE_BYTES:
                    raise ValueError('Artifact exceeds observation budget')
                content = path.read_bytes()
                sha = hashlib.sha256(content).hexdigest()
                if sha != record.get('sha256'):
                    raise ValueError('Artifact hash changed')
            value = {'path': name, 'bytes': len(content), 'suffix': Path(name).suffix,
                     'note': 'Metadata only; no per-row targets or predictions exposed.'}
            if Path(name).suffix.lower() in {'.npy', '.npz'}:
                from src.investigation_artifacts import numpy_artifact_metadata
                value.update(numpy_artifact_metadata(content, Path(name).suffix.lower()))
            elif name.endswith('.json'):
                obj = json.loads(content)
                value['shape'] = 'mapping' if isinstance(obj, dict) else 'list' if isinstance(obj, list) else 'scalar'
                if isinstance(obj, dict):
                    value['keys'] = list(obj)[:100]
                    value['collection_lengths'] = {key: len(item) for key, item in obj.items()
                                                   if isinstance(item, (list, dict))}
                    value['protocol_metadata'] = {key: item for key, item in obj.items() if key in
                        {'unit', 'split', 'aggregation', 'runs', 'metric', 'dataset', 'method'}
                        and isinstance(item, (str, int, float, bool, type(None)))}
                elif isinstance(obj, list):
                    value['rows'] = len(obj)
            elif name.endswith(('.csv', '.tsv')):
                import csv
                import io
                rows = csv.reader(io.StringIO(content.decode('utf-8')), delimiter='\t' if name.endswith('.tsv') else ',')
                value['columns'] = next(rows, [])[:100]
                value['rows'] = sum(1 for _ in rows)
            evidence.append({'id': 'artifact_' + sha[:16], 'source': name, 'sha256': sha, 'value': value})
        return self._result('artifact_contract', 'completed', evidence, 'Observed public file identities and structural metadata.')

    def _author_contract(self, arguments):
        if set(arguments) != {'path', 'function'} or not isinstance(arguments['function'], str):
            raise ValueError('Expected an original test path and function')
        name, function = arguments['path'], arguments['function']
        if not isinstance(name, str) or not name.endswith('.py'):
            raise ValueError('Original contract must be Python source')
        if not self.readable_source(name, (self.packet.get('file_inventory') or {}).get(name)):
            raise ValueError('Only public readable frozen contracts can be selected')
        content, sha = self._source(name)
        source = content.decode('utf-8')
        scope = 'original_frozen_source'
        candidate_sha = None
        if name in self._candidate_sources:
            candidate = self._candidate_sources[name]
            if len(candidate.encode('utf-8')) > 65536:
                raise ValueError('Candidate contract exceeds observation budget')
            def selected(text):
                return [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)
                        and node.name == function]
            original_test, candidate_test = selected(source), selected(candidate)
            if (len(original_test) != 1 or len(candidate_test) != 1
                    or ast.dump(original_test[0]) != ast.dump(candidate_test[0])):
                raise ValueError('Original selected assertion contract must be unchanged')
            source, scope = candidate, 'candidate_with_unchanged_original_assertions'
            candidate_sha = hashlib.sha256(candidate.encode('utf-8')).hexdigest()
        # The helper accepts only a very small pure AST language. It cannot
        # import author modules, perform I/O or run arbitrary original tests.
        result = self._child({'tool': 'author_contract', 'source': source, 'function': function})
        self._source(name)  # Recheck immutable source after the subprocess.
        return self._result('author_contract', result.get('status', 'rejected'), [
            {'id': 'contract_' + sha[:16] + '_' + function + ('_' + candidate_sha[:16] if candidate_sha else ''),
             'source': name, 'sha256': sha,
             'value': {'function': function, **result, 'scope': scope,
                       'candidate_sha256': candidate_sha, 'numerical_equivalence': 'unknown'}}],
            'Checked only the original selected pure-function assertion contract; scope is not whole-program equivalence.')

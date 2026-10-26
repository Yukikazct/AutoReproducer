"""Real-model development smoke test; invented fixture, never a paper holdout."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.audit_logger import AuditLogger
from src.intelligent_recovery import RecoverySession
from src.llm.llm_client import LLMClient
from src.repository_runner import RepositoryRunner
from src.source_repair import validate_compatibility_proposal
from src.author_self_repair import failure_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='https://api.deepseek.com')
    parser.add_argument('--model', default='deepseek-chat')
    parser.add_argument('--mode', choices=('single', 'collaborative'), default='collaborative')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--list-models', action='store_true')
    args = parser.parse_args()
    credential = os.environ.get('LLM_API_KEY') or getpass.getpass('API key (hidden, memory only): ')
    if args.list_models:
        endpoint = args.base_url.rstrip('/')
        if not endpoint.endswith('/v1'):
            endpoint += '/v1'
        request = urllib.request.Request(endpoint + '/models', headers={'Authorization': 'Bearer ' + credential})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        print(json.dumps({'models': [item['id'] for item in result.get('data', [])]}, ensure_ascii=False))
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required for the development smoke test')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    original, candidate = root / 'original', root / 'candidate'
    original.mkdir()
    sources = {'model.py': 'class Model:\n    def value(self):\n        return dict(answer=7).iteritems()\n',
               'train.py': 'from model import Model\nprint(list(Model().value())[0][1])\n',
               'README.md': 'Development compatibility fixture, not a real paper. The unchanged command is python train.py. '
                            'The program enumerates the original dictionary and prints its answer value. '
                            'Only Python API spelling may change; preserve the answer, data and computation.\n'}
    for name, source in sources.items():
        (original / name).write_text(source, encoding='utf-8')
    inventory = {name: {'sha256': hashlib.sha256((original / name).read_bytes()).hexdigest(), 'text_available': True}
                 for name in sources}
    shutil.copytree(original, candidate)
    plan = {'repository': {'url': 'https://github.com/invented-fixtures/python-api', 'revision': 'a' * 40},
            'steps': [{'id': 'train', 'kind': 'train', 'argv': ['python', 'train.py'], 'cwd': '.',
                       'entrypoint': {'kind': 'script', 'source_path': 'train.py'}}],
            'requirements': {'author': [], 'compatibility': []}}
    packet = {'file_inventory': inventory}
    environment = {'python_executable': sys.executable, 'requirements_txt': ''}
    logger = AuditLogger(log_dir=str(root / 'logs'), ledger_dir=str(root / 'ledger'))
    runner = RepositoryRunner(logger=logger)
    initial = runner.run(candidate, plan['steps'], environment)
    (root / 'initial_execution.json').write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding='utf-8')
    if initial.get('success') or 'iteritems' not in failure_diagnostic(initial):
        raise RuntimeError('Development fixture did not produce the preregistered real API failure')
    llm = LLMClient(base_url=args.base_url, model=args.model, api_key=credential, timeout=180)
    data = {}
    session = RecoverySession(llm, mode=args.mode)
    with session.scope():
        session.attach(data, root)
        task = session.investigate(trigger='execution_failure', plan=plan, packet=packet,
            workspace=original, environment=environment, execution=initial)
        if task and task['status'] == 'repair_proposed':
            result = runner.run_with_repairs(candidate, plan['steps'], environment, llm=llm,
                editable=['model.py', 'train.py'], initial_execution=initial,
                validate_patches=lambda workspace, files, failure:
                    validate_compatibility_proposal(workspace, files, failure_diagnostic(failure)))
        else:
            result = initial
        session.record_execution(result)
    unchanged = all(hashlib.sha256((original / name).read_bytes()).hexdigest() == record['sha256']
                    for name, record in inventory.items())
    passed = result.get('success') is True and (result.get('final') or {}).get('stdout', '').strip() == '7' and unchanged
    summary = {'origin': 'injected', 'split': 'dev', 'paper_reproduction': False,
               'model': args.model, 'mode': args.mode, 'passed': passed,
               'original_sources_unchanged': unchanged, 'numerical_equivalence': 'unknown',
               'investigation_status': session.state['status'], 'tasks': len(session.state['tasks']),
               'usage': session.state['usage'], 'execution_records': result.get('run_dir'),
               'note': 'Real API and owned Python execution on an invented compatibility fixture; not the 12-case holdout.'}
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

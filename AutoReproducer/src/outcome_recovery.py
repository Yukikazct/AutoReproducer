"""Restore a uniquely documented author invocation after a numerical failure.

This policy never trains, calls a model, edits source, or ranks settings by test
scores. A proposed command is a new experiment and must receive a fresh frozen
workspace/specification; the preceding experiment remains immutable.
"""
from copy import deepcopy
import hashlib
import itertools
import math
from pathlib import PurePosixPath
import re
import shlex

from src.author_experiment_selection import _entrypoint_cli, _name, dataset_cli_from_metadata
from src.author_runtime_dependencies import _terminal_repair_state
from src.execution_safety import cleanup_is_unconfirmed


_DESIGNATION = re.compile(r'\breproduc(?:e|ing|tion)\b.{0,180}\b(?:results?|paper|table)\b'
                          r'|\b(?:results?|paper|table)\b.{0,180}\breproduc(?:e|ing|tion)\b', re.I)
_VARIABLE = re.compile(r'\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))')
_IDENTITY = re.compile(r'^(?:model|method|arch|architecture|teacher|student|config|cfg|split|metric|fold|baseline|variant|algorithm|mode)(?:_|$)')
_MAX_FILES, _MAX_BYTES, _MAX_VARIANTS = 24, 1024 * 1024, 64


def _decision(status, reason, **extra):
    return {'version': 1, 'status': status, 'reason': reason, 'max_attempts': 1, **extra}


def _numerical_failure(validation):
    """Use the existing judgement, never invent another acceptance tolerance."""
    from src.reproduction_assessment import _pairing_problem
    assessment = validation.get('reproduction_assessment') or {}
    rows = (validation.get('author_claim_comparison') or {}).get('comparisons') or []
    if assessment.get('status') == 'completed' and assessment.get('source') == 'ai':
        if assessment.get('verdict') in {'reproduced', 'insufficient_evidence'}:
            return False
        verdicts = {item.get('actual_id'): item.get('verdict')
                    for item in assessment.get('metric_assessments', [])}
        for row in rows:
            gap = row.get('difference')
            if (_pairing_problem(row) or type(gap) not in (int, float) or not math.isfinite(gap)):
                continue
            verdict = verdicts.get((row.get('actual') or {}).get('id'))
            if verdict == 'not_reproduced':
                return True
            # Partial numerical agreement with known different settings is
            # still incomplete. A later independent documentation check must
            # prove a unique omitted-argument contract before any rerun.
            adverse = (row.get('direction') == 'higher' and gap < 0
                       or row.get('direction') == 'lower' and gap > 0)
            if (verdict == 'partially_reproduced' and adverse
                    and (row.get('settings') or {}).get('status') == 'different'):
                return True
        return False
    return any(row.get('status') == 'comparable' and row.get('meets_reference') is False
               and type(row.get('difference')) in (int, float) and math.isfinite(row['difference'])
               for row in rows)


def _arguments(cli, arguments):
    """Describe simple original argparse tokens, preserving repeated values."""
    if cli is None or any(item.startswith('<') for item in cli.get('missing', [])):
        return None
    declarations = cli.get('options', [])
    if any(item['nargs'] is not None or item['action'] not in {'store', 'store_true', 'store_false'}
           or any(not flag.startswith('-') for flag in item['flags']) for item in declarations):
        return None
    flags = {flag: item for item in declarations for flag in item['flags']}
    result, index = {}, 0
    while index < len(arguments):
        token = arguments[index]
        if token == '--':
            return None
        flag, equals, value = token.partition('=')
        item = flags.get(flag)
        if item is None:
            return None
        if item['action'] == 'store':
            if not equals:
                index += 1
                if index >= len(arguments) or arguments[index] in flags or arguments[index].startswith('--'):
                    return None
                value = arguments[index]
        elif equals:
            return None
        else:
            value = item['action'] == 'store_true'
        result.setdefault(item['dest'], []).append(value)
        index += 1
    return result


def _same_default(values, default):
    if len(values) != 1:
        return False
    value = values[0]
    if type(default) in (int, float) and isinstance(value, str):
        try:
            return math.isfinite(float(value)) and float(value) == default
        except (ValueError, OverflowError):
            return False
    return value == default or isinstance(value, str) and value == str(default)


def _retained_cpu_flags(cli, original_options, options, original_arguments, device, defaults):
    """Keep an already frozen CPU switch absent from the paper launcher.

    This never introduces a hardware option. Only the original literal boolean
    CPU flag can accompany the documented command; all other original options
    must still appear with their exact values in that command.
    """
    if device != 'cpu':
        return []
    retained = []
    for name, flag in (('no_cuda', '--no-cuda'), ('cpu', '--cpu')):
        if name not in original_options or name in options:
            continue
        declarations = [item for item in cli['options'] if item['dest'] == name]
        default_records = [item for item in cli['defaults'] if item['dest'] == name]
        if len(declarations) != 1 or len(default_records) != 1:
            continue
        declaration = declarations[0]
        if (original_options[name] == [True] and defaults.get(name) is False
                and default_records[0]['value'] is False
                and declaration.get('action') == 'store_true' and declaration.get('nargs') is None
                and flag in declaration.get('flags', []) and original_arguments.count(flag) == 1):
            retained.append(flag)
    return retained


def _logical_lines(text):
    pending, start = [], 0
    for number, line in enumerate(text.splitlines(), 1):
        if not pending:
            start = number
        pending.append(line)
        if line.rstrip().endswith('\\'):
            continue
        quote = '\n'.join(pending)
        yield start, quote, re.sub(r'\\\r?\n\s*', ' ', quote).strip().removeprefix('$ ').strip('`').strip()
        pending = []


def _designated_sources(packet, workspace):
    """Only README-designated reproduction blocks or linked fixed launchers."""
    from src.discovered_repository_plan import _load_execution_source
    inventory = packet['file_inventory']
    readmes = sorted(name for name in inventory if PurePosixPath(name).name.lower().startswith('readme'))
    examined, consumed, sources = 0, 0, []
    for name in readmes[:_MAX_FILES]:
        size = inventory[name].get('bytes', 0)
        consumed += size
        if consumed > _MAX_BYTES:
            break
        source = _load_execution_source(packet, workspace, name)
        lines = source['text'].splitlines()
        for index, line in enumerate(lines):
            if not _DESIGNATION.search(line):
                continue
            end = min(len(lines), index + 21)
            for next_index in range(index + 1, end):
                if re.match(r'^\s*#{1,6}\s', lines[next_index]):
                    end = next_index
                    break
            block = '\n' * index + '\n'.join(lines[index:end])
            designation = {'path': name, 'sha256': source['sha256'], 'line': index + 1, 'quote': line}
            sources.append((name, block, source['sha256'], designation, '.'))
            context = '\n'.join(lines[index:end])
            paths = set(re.findall(r'(?:[\w.-]+/)*[\w.-]+\.sh\b', context))
            # A named directory is accepted only in the same designation line.
            for directory in re.findall(r'`([\w./-]+/)`', line):
                paths.update(path for path in inventory if path.startswith(directory) and path.endswith('.sh'))
            for path in sorted(paths):
                if path not in inventory:
                    continue
                examined += 1
                consumed += inventory[path].get('bytes', 0)
                if examined > _MAX_FILES or consumed > _MAX_BYTES:
                    raise ValueError('Documented reproduction source reading exceeds its bound')
                launcher = _load_execution_source(packet, workspace, path)
                sources.append((path, launcher['text'], launcher['sha256'], designation, '.'))
    return sources


def _literal_commands(sources):
    for name, text, sha, designation, initial_cwd in sources:
        choices, cwd = {}, initial_cwd
        for number, quote, line in _logical_lines(text):
            loop = re.fullmatch(r'for\s+([A-Za-z_]\w*)\s+in\s+(.+?)(?:;\s*do)?', line)
            if loop:
                values = shlex.split(loop[2], comments=True)
                if not values or any(re.search(r'[$`<>;|&()]', value) for value in values):
                    choices[loop[1]] = []
                else:
                    choices[loop[1]] = values
                continue
            if line == 'done':
                choices.clear()
                continue
            if re.fullmatch(r'cd\s+[\w./-]+', line):
                cwd = line.split(maxsplit=1)[1]
                continue
            if not re.match(r'^python(?:3)?\s+', line):
                continue
            tokens = shlex.split(line, comments=True)
            for index, token in enumerate(tokens):
                if token in {'>', '>>', '2>', '2>>'}:
                    tokens = tokens[:index]
                    break
            if any(re.search(r'[`<>;|&()]', token) for token in tokens):
                continue
            variables = list(dict.fromkeys(match[1] or match[2] for token in tokens for match in _VARIABLE.finditer(token)))
            counts = [len(choices.get(variable, [])) for variable in variables]
            if variables and (not all(counts) or math.prod(counts) > _MAX_VARIANTS):
                continue
            variants = itertools.product(*(choices[variable] for variable in variables)) if variables else [()]
            for values in variants:
                bindings = dict(zip(variables, values))
                argv = [_VARIABLE.sub(lambda match: bindings[match[1] or match[2]], token) for token in tokens]
                if any('$' in token for token in argv):
                    continue
                yield argv, cwd, {'path': name, 'sha256': sha, 'line': number, 'quote': quote,
                                   'template_bindings': bindings, 'designation': designation}


def prepare_outcome_recovery(plan, packet, workspace, execution, validation, request):
    """Propose one new frozen author experiment, or explain why none is safe.

    Runtime owns the independent child directory, source/data copy, cancellation,
    environment lifecycle and one-attempt budget. Returned plan/packet are copies.
    """
    if request.get('source_repair') is not True or request.get('source_repair_auto') is not True:
        return _decision('not_requested', 'Automatic recovery is disabled; no candidate source was read.')
    if not _numerical_failure(validation):
        return _decision('not_needed', 'No established incomplete numerical outcome; missing evidence alone does not authorize retraining.')
    if (plan.get('scope') != 'author_code_execution' or request.get('strict_validation') is True
            or (validation.get('strict_validation') or {}).get('requested')
            or validation.get('source_modified') or (validation.get('author_claim_comparison') or {}).get('source_modified')):
        return _decision('unresolved', 'Strict or source-modified execution cannot use original-command outcome recovery.')
    if request.get('_outcome_recovery_attempt', 0):
        return _decision('unresolved', 'The single independent outcome recovery attempt has already been used.')
    from src.discovered_repository_plan import _canonical
    if (plan.get('repository') != {key: packet.get('repository', {}).get(key) for key in ('url', 'revision')}
            or plan.get('pdf_sha256') != (packet.get('pdf') or {}).get('sha256')
            or plan.get('evidence_sha256') != hashlib.sha256(_canonical(packet).encode('utf-8')).hexdigest()):
        return _decision('unresolved', 'The PDF, repository or source packet differs from the original frozen experiment.')
    terminal = _terminal_repair_state(execution, execution.get('effective_env_config') or {})
    checks = validation.get('completion_checks') or {}
    receipts = validation.get('author_execution') or []
    if (execution.get('success') is not True or terminal or cleanup_is_unconfirmed(execution)
            or checks.get('owned_processes_finished') is not True
            or checks.get('all_required_author_commands_returned') is not True
            or not receipts or any(item.get('natural_return') is not True for item in receipts)):
        return _decision('unresolved', 'Natural completion and confirmed process cleanup are required before a new attempt.')
    steps = plan.get('steps') or []
    if len(steps) != 1 or steps[0].get('kind') != 'train' or steps[0].get('entrypoint', {}).get('kind') not in {'script', 'module'}:
        return _decision('unresolved', 'No supported single-stage original command omission contract is available.')
    from src.author_entrypoints import resolve_entrypoint
    from src.author_execution_plan import validate_author_execution_plan
    working = deepcopy(packet)
    original = steps[0]
    evidence, candidates = [], {}
    try:
        from src.author_comparison_context import frozen_cli_inputs
        if frozen_cli_inputs(working, plan, workspace)['status'] != 'observed':
            return _decision('unresolved', 'The original parser inputs and default binding order are not uniquely static.')
        cli, parser_source = _entrypoint_cli(working, workspace, original)
        bare = {**original, 'entrypoint': {key: original['entrypoint'][key] for key in ('kind', 'target')}}
        _, _, original_arguments = resolve_entrypoint(bare, working, workspace)
        original_options = _arguments(cli, original_arguments)
        original_dataset = dataset_cli_from_metadata(cli, original_arguments)
        if original_options is None or original_dataset['status'] != 'observed_cli_selection':
            return _decision('unresolved', 'The original command has no unambiguous static argument and dataset contract.')
        defaults = {item['dest']: item['value'] for item in cli['defaults']}
        for argv, cwd, source in _literal_commands(_designated_sources(working, workspace)):
            if cwd != original['cwd'] or argv == original['argv']:
                continue
            entrypoint, _, arguments = resolve_entrypoint({'argv': argv, 'cwd': cwd}, working, workspace)
            if entrypoint != original['entrypoint'] or argv[:len(argv) - len(arguments)] != original['argv'][:len(original['argv']) - len(original_arguments)]:
                continue
            options = _arguments(cli, arguments)
            selected = dataset_cli_from_metadata(cli, arguments)
            if options is None or selected['status'] != 'observed_cli_selection' or _name(selected['value']) != _name(original_dataset['value']):
                continue
            retained = _retained_cpu_flags(cli, original_options, options, original_arguments,
                                           plan['capture']['device'], defaults)
            if retained:
                argv, arguments = [*argv, *retained], [*arguments, *retained]
                options = _arguments(cli, arguments)
                source = {**source, 'retained_original_cpu_flags': retained,
                    'retained_cpu_flag_evidence': {'parser': deepcopy(parser_source),
                        'declarations': [deepcopy(item) for item in cli['options']
                                         if any(flag in item['flags'] for flag in retained)],
                        'defaults': [deepcopy(item) for item in cli['defaults']
                                     if item['dest'] in {'no_cuda', 'cpu'}]}}
            if any(options.get(name) != values for name, values in original_options.items()):
                continue  # Never overwrite even one original explicit setting.
            added = set(options) - set(original_options)
            if not added or any(_IDENTITY.match(name) for name in added):
                continue
            if any(re.search(r'^(?:data_(?:root|path|dir|file)|(?:train|test|val|validation)_(?:path|file|split))$', name)
                   for name in added):
                continue
            if any('seed' in name and options[name] != [str(defaults.get(name))] for name in added):
                continue
            if all(name in defaults and _same_default(options[name], defaults[name]) for name in added):
                continue  # Explicitly repeating defaults cannot explain a numerical discrepancy.
            # Preserve all original literal tokens as well as their parsed values.
            cursor = iter(argv)
            if not all(any(value == token for value in cursor) for token in original['argv']):
                continue
            proposed_step = {key: deepcopy(value) for key, value in original.items()
                             if key in {'id', 'kind', 'depends_on', 'timeout_s', 'required'}}
            proposed_step.update(argv=argv, cwd=cwd)
            payload = {'version': 1, 'scope': plan['scope'], 'repository': deepcopy(plan['repository']),
                'device': plan['capture']['device'], 'citations': deepcopy(plan.get('citations', [])),
                'claims': deepcopy(plan.get('claims', {})), 'datasets': deepcopy(plan.get('datasets', [])),
                'requirements': deepcopy(plan.get('requirements', {})), 'steps': [proposed_step]}
            validated = validate_author_execution_plan(payload, working, workspace)
            validated['command_source'] = deepcopy(source)
            key = (cwd, tuple(argv))
            candidates[key] = (payload, source)
            evidence.append({'command': source, 'added_options': sorted(added), 'parser': parser_source})
        if len(candidates) != 1:
            return _decision('unresolved', 'No unique README-designated reproduction command only fills omitted original parameters.',
                             candidate_count=len(candidates), evidence=evidence)
        payload, source = next(iter(candidates.values()))
        # Source reads changed the packet: recompute the plan against its final form.
        recovered = validate_author_execution_plan(payload, working, workspace)
        recovered['command_source'] = deepcopy(source)
        return _decision('proposed', 'A unique fixed author reproduction command supplies omitted parameters; run it once as a new experiment.',
                         plan=recovered, packet=working, evidence=evidence,
                         original_commands=deepcopy(steps), original_plan_preserved=True,
                         selection_basis='author_documentation_only', tolerance_changed=False)
    except (ValueError, TypeError, KeyError, OSError, SyntaxError) as exc:
        return _decision('unresolved', 'Author command recovery evidence could not be verified: ' + str(exc), evidence=evidence)

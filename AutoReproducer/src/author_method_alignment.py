"""Narrow static component bridges; never canonicalize or rename a method.

This is source evidence for a reference-only comparison, not verification that
Python executed the inferred branch or that two experimental protocols match.
No repository code is imported or evaluated.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re

from src.discovered_repository_plan import _read_source


_NAME = r'[A-Za-z][A-Za-z0-9_+.-]*'
_RELATION = re.compile(
    r'(?<![\w+.-])(?P<wrapper>' + _NAME + r')\s*\(\s*applied\s+to\s+'
    r'(?P<base>' + _NAME + r')\s*,\s*i\.e\.\s*,?\s*'
    r'(?P<label>' + _NAME + r')\s*\)', re.I)


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _attribute(node, owner='self'):
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == owner):
        return node.attr
    return None


def _one(values, reason):
    if len(values) != 1:
        raise ValueError(reason)
    return values[0]


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _location(path, source, node):
    return {'path': path, 'sha256': source['sha256'], 'line': node.lineno,
            'quote': ast.get_source_segment(source['text'], node)}


def _source(packet, workspace, path):
    entry = packet['repository']['files'][path]
    expected = entry if isinstance(entry, str) else entry['sha256']
    source = _read_source(workspace, path, expected)
    if isinstance(entry, dict) and entry.get('text', source['text']) != source['text']:
        raise ValueError('Frozen source text differs from its verified bytes')
    return source, ast.parse(source['text'])


def _declaration_bindings(tree, argv, selector):
    """Bind one explicit string option, without accepting other parser defaults."""
    declarations = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and isinstance(node.func.value, ast.Name)):
            continue
        flags = [_literal(arg) for arg in node.args]
        if not flags or not all(isinstance(flag, str) and flag.startswith('-') for flag in flags):
            continue
        options = {kw.arg: kw.value for kw in node.keywords}
        dest = _literal(options['dest']) if 'dest' in options else max(flags, key=len).lstrip('-').replace('-', '_')
        if dest == selector:
            declarations.append((node, flags, options))
    node, flags, options = _one(declarations, 'Selector has no unique argparse declaration')
    if any(key in options for key in ('nargs', 'const')) or any(kw.arg is None for kw in node.keywords):
        raise ValueError('Selector uses a non-scalar argparse declaration')
    if 'action' in options and _literal(options['action']) != 'store':
        raise ValueError('Selector does not store its explicit string input')
    if 'type' in options and not (isinstance(options['type'], ast.Name) and options['type'].id == 'str'):
        raise ValueError('Selector conversion is not the builtin string conversion')
    for other in ast.walk(tree):
        if (isinstance(other, ast.Name) and other.id == 'str' and isinstance(other.ctx, (ast.Store, ast.Del))
                or isinstance(other, (ast.FunctionDef, ast.ClassDef)) and other.name == 'str'):
            raise ValueError('Selector string converter is rebound')
        if isinstance(other, ast.Call) and isinstance(other.func, ast.Attribute) and other.func.attr == 'set_defaults':
            raise ValueError('Parser mutations have no component binding contract')
    found = []
    for index, token in enumerate(argv):
        flag, equals, value = token.partition('=')
        if flag in flags:
            if not equals:
                if index + 1 >= len(argv) or argv[index + 1].startswith('-'):
                    raise ValueError('Selector lacks its explicit scalar value')
                value = argv[index + 1]
            found.append((value, [token] if equals else [token, value]))
        elif flag.startswith('-') and any(candidate.startswith(flag) for candidate in flags):
            raise ValueError('Abbreviated selector input is ambiguous')
    value, tokens = _one(found, 'Selector needs exactly one explicit argv value')
    if '--' in argv[:argv.index(tokens[0]) + 1]:
        raise ValueError('Selector appears after the argparse terminator')
    if 'choices' in options:
        choices = _literal(options['choices'])
        if not isinstance(choices, (list, tuple)) or value not in choices:
            raise ValueError('Selector choices are not a matching literal collection')
    return value, node, tokens


def _argument_flow(tree, constructor, selector):
    """One parse_args object may enter one containing function unchanged."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    owner = constructor
    while owner in parents and not isinstance(owner, ast.FunctionDef):
        owner = parents[owner]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'parse_args'
             and isinstance(node.value.func.value, ast.Name) and not node.value.args and not node.value.keywords]
    parsed = _one(calls, 'No unique original parse_args binding')
    variable, parser = parsed.targets[0].id, parsed.value.func.value.id
    if isinstance(owner, ast.FunctionDef):
        positions = [i for i, arg in enumerate(owner.args.args) if arg.arg == variable]
        position = _one(positions, 'Constructor function does not receive the parsed input directly')
        invocations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == owner.name]
        call = _one(invocations, 'Constructor function has ambiguous invocation paths')
        if len(call.args) <= position or not isinstance(call.args[position], ast.Name) or call.args[position].id != variable:
            raise ValueError('Constructor receives another argument object')
    else:
        call = None
    parser_creations = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == parser for target in node.targets)
                        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                        and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'argparse'
                        and node.value.func.attr == 'ArgumentParser']
    _one(parser_creations, 'No unique direct argparse parser construction')
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == variable and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node is not parsed.targets[0]:
                raise ValueError('Parsed argument object is rebound')
        if _attribute(node, variable) and isinstance(node.ctx, (ast.Store, ast.Del)):
            raise ValueError('Parsed argument fields are overridden')
        if _attribute(node, variable) == '__dict__':
            raise ValueError('Parsed argument storage is accessed dynamically')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == variable:
                raise ValueError('Parsed argument object has an opaque method call')
            if any(isinstance(arg, ast.Name) and arg.id == variable for arg in node.args) and node is not call:
                raise ValueError('Parsed argument object escapes the supported direct call')
    return variable, parser, parsed


def _test_value(node, parameter):
    if (isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1
            and isinstance(node.ops[0], ast.Eq) and isinstance(node.left, ast.Name)
            and node.left.id == parameter and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, str)):
        return node.comparators[0].value
    return None


def _reachable_calls(function, required):
    """Check source data dependencies, not whether a loop executed at runtime."""
    returns = []

    def expression(node, env):
        flags = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Name):
                flags.update(env.get(item.id, ()))
            if isinstance(item, ast.Call):
                attr = _attribute(item.func)
                if attr in required:
                    flags.add(attr)
        return flags

    def statements(body, env):
        env = {key: set(value) for key, value in env.items()}
        for node in body:
            if isinstance(node, ast.Assign):
                flags = expression(node.value, env)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = flags
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
                env[node.target.id] = expression(node.value, env)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                env[node.target.id] = env.get(node.target.id, set()) | expression(node.value, env)
            elif isinstance(node, ast.If):
                left, right = statements(node.body, env), statements(node.orelse, env)
                env = {key: left.get(key, set()) & right.get(key, set()) for key in left.keys() | right.keys()}
            elif isinstance(node, ast.For):
                # Deliberately potential source flow only; a zero-iteration loop
                # is one reason this evidence can never establish runtime use.
                env = statements(node.body, env)
            elif isinstance(node, ast.Return):
                returns.append(expression(node.value, env) if node.value else set())
                return env
            elif isinstance(node, (ast.Try, ast.While, ast.With, ast.Match, ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError('Method forward has unsupported control flow')
        return env

    statements(function.body, {})
    return bool(returns) and all(set(required) <= flags for flags in returns)


def _class_bridge(tree, implementation, wrapper, base, constructor):
    classes = {name: _one([node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name],
                         'Component class is absent or ambiguous') for name in (implementation, wrapper)}
    model, wrapped = classes[implementation], classes[wrapper]
    if any(item.decorator_list or item.keywords for item in classes.values()):
        raise ValueError('Decorated or metaclass component definitions are unsupported')
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in classes and isinstance(node.ctx, (ast.Store, ast.Del)):
            raise ValueError('Component class name is rebound')
        if isinstance(node, ast.alias) and (node.asname or node.name) in classes:
            raise ValueError('Component class name is shadowed by an import')
        if isinstance(node, ast.FunctionDef) and node.name in {'__getattr__', '__getattribute__', '__setattr__'}:
            raise ValueError('Component uses dynamic instance attribute semantics')
    init = _one([node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == '__init__'],
                'Implementation needs one constructor')
    forward = _one([node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'forward'],
                   'Implementation needs one forward')
    parameters = init.args.args[1:]
    bindings = {parameters[i].arg: arg for i, arg in enumerate(constructor.args) if i < len(parameters)}
    if len(constructor.args) > len(parameters) or init.args.vararg or init.args.kwarg or any(kw.arg is None for kw in constructor.keywords):
        raise ValueError('Implementation uses variadic constructor binding')
    for kw in constructor.keywords:
        if kw.arg in bindings:
            raise ValueError('Constructor argument has multiple bindings')
        bindings[kw.arg] = kw.value
    choices = []
    for parameter, argument in bindings.items():
        if not (isinstance(argument, ast.Attribute) and isinstance(argument.value, ast.Name)):
            continue
        branches = [node for node in ast.walk(init) if isinstance(node, ast.If) and _test_value(node.test, parameter) == base]
        if len(branches) == 1:
            choices.append((parameter, argument, branches[0]))
    parameter, argument, branch = _one(choices, 'No unique constructor selector branch for the paper component')
    if any(isinstance(node, ast.Name) and node.id == parameter and isinstance(node.ctx, (ast.Store, ast.Del)) for node in ast.walk(init)):
        raise ValueError('Constructor selector is overridden')
    base_attributes = {}
    for node in ast.walk(ast.Module(body=branch.body, type_ignores=[])):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and _attribute(node.targets[0])
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)):
            attr = _attribute(node.targets[0])
            if attr in base_attributes:
                raise ValueError('Selected base attribute has multiple definitions')
            base_attributes[attr] = (node.value.func.id, node)
    if not base_attributes:
        raise ValueError('Selected branch constructs no direct base component')
    base_constructor = _one(list({item[0] for item in base_attributes.values()}), 'Selected branch uses different base constructors')
    wrappers = []
    for node in ast.walk(init):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == wrapper):
            if not node.args or _attribute(node.args[0]) not in base_attributes:
                raise ValueError('Wrapper does not consume the selected base component')
            assignments = [item for item in ast.walk(init) if isinstance(item, ast.Assign) and item.value is node
                           and len(item.targets) == 1 and _attribute(item.targets[0])]
            assignment = _one(assignments, 'Wrapper lacks a direct instance attribute binding')
            wrappers.append((node, assignment, _attribute(assignment.targets[0])))
    wrapper_attribute = _one(list({row[2] for row in wrappers}), 'No unique wrapper attribute')
    parents = {child: node for node in ast.walk(init) for child in ast.iter_child_nodes(node)}
    chain = branch
    while (chain in parents and isinstance(parents[chain], ast.If)
           and parents[chain].orelse == [chain] and _test_value(parents[chain].test, parameter) is not None):
        chain = parents[chain]
    allowed_base_nodes = set(ast.walk(chain))
    allowed_wrapper_assignments = {item[1] for item in wrappers}
    for node in ast.walk(init):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                attr = _attribute(target)
                if (attr in base_attributes and node not in allowed_base_nodes
                        or attr == wrapper_attribute and node not in allowed_wrapper_assignments):
                    raise ValueError('Component attribute is overwritten outside its selected construction')
    if not _reachable_calls(forward, {wrapper_attribute}):
        raise ValueError('Wrapper output does not reach the implementation result')
    direct_used = [attr for attr in base_attributes if _reachable_calls(forward, {attr, wrapper_attribute})]
    if not direct_used:
        raise ValueError('Selected base is not used in the implementation result')
    wrapper_init = _one([node for node in wrapped.body if isinstance(node, ast.FunctionDef) and node.name == '__init__'],
                        'Wrapper lacks a unique constructor')
    wrapper_forward = _one([node for node in wrapped.body if isinstance(node, ast.FunctionDef) and node.name == 'forward'],
                           'Wrapper lacks a unique forward')
    if len(wrapper_init.args.args) < 2:
        raise ValueError('Wrapper has no first component argument')
    first = wrapper_init.args.args[1].arg
    component_attrs = [(_attribute(node.targets[0]), node) for node in ast.walk(wrapper_init)
                       if isinstance(node, ast.Assign) and len(node.targets) == 1 and _attribute(node.targets[0])
                       and isinstance(node.value, ast.Name) and node.value.id == first]
    component_attr, component_assignment = _one(component_attrs, 'Wrapper does not retain its component directly')
    writes = [node for node in ast.walk(wrapper_init) if isinstance(node, ast.Attribute)
              and _attribute(node) == component_attr and isinstance(node.ctx, (ast.Store, ast.Del))]
    if len(writes) != 1:
        raise ValueError('Wrapper component attribute is overwritten')
    if not _reachable_calls(wrapper_forward, {component_attr}):
        raise ValueError('Wrapper component does not reach the wrapper result')
    for method in (forward, wrapper_forward):
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                raise ValueError('Forward mutates component attributes')
            if isinstance(node, ast.Call) and any(isinstance(arg, ast.Name) and arg.id == 'self' for arg in node.args):
                raise ValueError('Forward passes its instance to an opaque call')
    return {'selector': argument.attr, 'argument_owner': argument.value.id,
            'parameter': parameter, 'branch': branch, 'base_constructor': base_constructor,
            'base_definitions': [item[1] for item in base_attributes.values()],
            'wrappers': [item[1] for item in wrappers], 'forward': forward,
            'wrapper_forward': wrapper_forward, 'component_assignment': component_assignment}


def align_method_components(reference, actual, packet, plan, workspace, record):
    """Return a source-bound static bridge or unknown, preserving every label.

    Supported paper syntax is deliberately narrow: ``W (applied to B, i.e.,
    W-B)`` on one hashed PDF page. The selected implementation must explicitly
    receive B from the current argv, construct it in an enum branch, and use
    both it and W in the returned forward data flow. All other forms stay
    unknown. Callers must cap a successful comparison at reference_only.
    """
    result = {'status': 'unknown', 'reference_label': reference.get('method'),
              'actual_label': actual.get('method'), 'max_comparison_status': 'reference_only',
              'independently_verified': False, 'effective_runtime_verified': False}
    try:
        implementation, label = actual['method'], reference['method']
        if not implementation.isidentifier() or not isinstance(plan, dict):
            raise ValueError('No supported literal implementation label or frozen plan')
        step_id, metric_id = record['step_id'], record['id']
        if not isinstance(step_id, str) or not isinstance(metric_id, str) or not metric_id.startswith(step_id + ':'):
            raise ValueError('Metric does not bind to one original execution step')
        if actual.get('id', metric_id) != metric_id or actual.get('step_id', step_id) != step_id:
            raise ValueError('Actual context refers to another metric or step')
        if record.get('method') and record['method'] != implementation:
            raise ValueError('Observed method differs from the literal implementation')
        step = _one([item for item in plan['steps'] if item['id'] == step_id], 'Metric step is missing or ambiguous')
        argv = step['argv']
        if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv) or len(argv) < 2:
            raise ValueError('Unsupported original argv')
        entrypoint = step['entrypoint']
        path = entrypoint['source_path']
        target = entrypoint['target']
        if entrypoint['kind'] != 'script' or argv[1] != target:
            raise ValueError('Only a direct frozen Python script entrypoint is supported')
        if (PurePosixPath(step.get('cwd') or '.') / target).as_posix() != path:
            raise ValueError('Entrypoint source does not match its original cwd and argv')
        revision = packet['repository']['revision']
        if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Repository revision is not immutable')
        for node in (record, actual, plan):
            for key in ('repository_revision', 'revision'):
                if node.get(key, revision) != revision:
                    raise ValueError('Metric, context or plan refers to another repository revision')
        for citation in record.get('interpretation_citations', []):
            if citation.get('source') == 'repository' and citation.get('revision', revision) != revision:
                raise ValueError('Metric interpretation belongs to another repository revision')
        pdf_hash = packet['pdf']['sha256']
        if not re.fullmatch(r'[0-9a-f]{64}', pdf_hash):
            raise ValueError('PDF identity is not immutable')
        if reference.get('pdf_sha256', pdf_hash) != pdf_hash:
            raise ValueError('Reference belongs to another PDF identity')
        relations = []
        for page in packet['pdf']['pages']:
            if _digest(page['text']) != page['sha256']:
                raise ValueError('Frozen PDF page text changed')
            for match in _RELATION.finditer(page['text']):
                if match['label'] == label and label == match['wrapper'] + '-' + match['base']:
                    relations.append((page, match))
        page, relation = _one(relations, 'No unique explicit paper component naming relation')
        source, tree = _source(packet, workspace, path)
        summary = record.get('summary_source')
        if summary and (summary.get('path') != path or summary.get('source_sha256') != source['sha256']):
            raise ValueError('Metric summary belongs to another frozen source')
        constructor = _one([node for node in ast.walk(tree) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name) and node.func.id == implementation],
                           'No unique direct implementation constructor call')
        modules = []
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module or '.' in node.module:
                continue
            if any(alias.name == '*' or alias.name == implementation and alias.asname is None for alias in node.names):
                candidate = (PurePosixPath(path).parent / (node.module + '.py')).as_posix()
                if candidate in packet['repository']['files']:
                    modules.append(candidate)
        model_path = _one(modules, 'Implementation import is not a unique local module')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if (alias.asname or alias.name) == implementation and not (
                            isinstance(node, ast.ImportFrom) and node.module + '.py' == PurePosixPath(model_path).name
                            and alias.name == implementation and alias.asname is None):
                        raise ValueError('Implementation import binding is shadowed')
        model_source, model_tree = _source(packet, workspace, model_path)
        bridge = _class_bridge(model_tree, implementation, relation['wrapper'], relation['base'], constructor)
        argument_owner, parser, parsed = _argument_flow(tree, constructor, bridge['selector'])
        if bridge['argument_owner'] != argument_owner:
            raise ValueError('Constructor selector uses another argument object')
        value, declaration, tokens = _declaration_bindings(tree, argv[2:], bridge['selector'])
        if declaration.func.value.id != parser or value != relation['base']:
            raise ValueError('Explicit selector does not select the paper base component')
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == implementation and isinstance(node.ctx, (ast.Store, ast.Del)):
                raise ValueError('Imported implementation name is overridden')
        evidence = [_location(path, source, item) for item in (parsed, declaration, constructor)]
        evidence += [_location(model_path, model_source, item) for item in (
            bridge['branch'], *bridge['base_definitions'], *bridge['wrappers'],
            bridge['component_assignment'], bridge['forward'], bridge['wrapper_forward'])]
        observed = {key: deepcopy(record.get(key)) for key in ('id', 'step_id', 'name', 'split', 'value', 'source', 'source_quote')}
        result.update(status='source_bound_static', reason='Explicit paper naming and unchanged source component selection agree; static reference-only evidence.',
            binding={'metric_id': metric_id, 'step_id': step_id, 'metric_sha256': _digest(json.dumps(observed, sort_keys=True, ensure_ascii=False)),
                     'argv': deepcopy(argv), 'cwd': step.get('cwd'), 'entrypoint': deepcopy(entrypoint),
                     'repository_url': packet['repository'].get('url'), 'repository_revision': revision,
                     'pdf_sha256': pdf_hash, 'source_sha256': {path: source['sha256'], model_path: model_source['sha256']}},
            components={'wrapper': relation['wrapper'], 'base': relation['base'], 'implementation': implementation,
                        'selector': bridge['selector'], 'parameter': bridge['parameter'], 'explicit_tokens': tokens,
                        'base_constructor': bridge['base_constructor']},
            paper_evidence={'page': page['page'], 'sha256': page['sha256'], 'pdf_sha256': pdf_hash,
                            'quote': relation.group(), 'start': relation.start(), 'end': relation.end()},
            source_evidence=evidence,
            limitations=['Static source selection and potential forward data flow only; runtime-effective configuration is not independently verified.',
                         'No protocol, hyperparameter, ablation or numerical equivalence is established; this bridge can never permit comparable or strict acceptance.'])
    except (ValueError, TypeError, KeyError, OSError, SyntaxError, UnicodeError, AttributeError, IndexError) as error:
        result['reason'] = str(error)
    return result

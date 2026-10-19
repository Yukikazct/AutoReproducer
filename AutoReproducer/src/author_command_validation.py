"""Conservative checks for literal argparse options before environment setup."""
import ast
import re


def completes_required_options(arguments, original, missing):
    """A launcher completion may fill missing options, never change the other argv."""
    allowed = {flag for group in missing for flag in group.split('/')}
    retained = []
    index = 0
    while index < len(arguments):
        flag, separator, _ = arguments[index].partition('=')
        if flag in allowed:
            index += 1
            if not separator and index < len(arguments) and (
                    not arguments[index].startswith('-') or re.fullmatch(r'-\d+(?:\.\d+)?', arguments[index])):
                index += 1
        else:
            retained.append(arguments[index])
            index += 1
    return retained == original


def missing_required_options(source, arguments):
    """Check only an invoked, plainly constructed parser; never invent values.

    Dynamic/custom parsers, functions and unknown branches remain the author's
    runtime responsibility. Unrelated parsers must not block a valid command.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    modules, constructors, parsers = set(), set(), {}
    missing = []

    def statements(body):
        for statement in body:
            if (isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare)
                    and isinstance(statement.test.left, ast.Name) and statement.test.left.id == '__name__'
                    and len(statement.test.ops) == 1 and isinstance(statement.test.ops[0], ast.Eq)
                    and len(statement.test.comparators) == 1
                    and isinstance(statement.test.comparators[0], ast.Constant)
                    and statement.test.comparators[0].value == '__main__'):
                yield from statements(statement.body)
            else:
                yield statement

    for node in statements(tree.body):
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == 'argparse')
        elif isinstance(node, ast.ImportFrom) and node.module == 'argparse' and node.level == 0:
            constructors.update(alias.asname or alias.name for alias in node.names if alias.name == 'ArgumentParser')
        value = node.value if isinstance(node, (ast.Assign, ast.Expr, ast.AnnAssign)) else None
        target = (node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name) else None)
        if target:
            parsers.pop(target, None)
            modules.discard(target)
            constructors.discard(target)
        if isinstance(node, ast.Assign):
            for assigned in node.targets:
                if isinstance(assigned, ast.Attribute) and isinstance(assigned.value, ast.Name):
                    parsers.pop(assigned.value.id, None)
                    modules.discard(assigned.value.id)
        if not isinstance(value, ast.Call):
            if isinstance(value, ast.Name):
                parsers.pop(value.id, None)  # An untracked alias could mutate the parser.
            # Unknown control flow may mutate the parser or choose another argv.
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                touched = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
                for name in touched:
                    parsers.pop(name, None)
            continue
        function = value.func
        constructor = (isinstance(function, ast.Name) and function.id in constructors
                       or isinstance(function, ast.Attribute) and function.attr == 'ArgumentParser'
                       and isinstance(function.value, ast.Name) and function.value.id in modules)
        if target and constructor:
            keywords = {item.arg: item.value for item in value.keywords}
            if (value.args or any(name not in {'prog', 'description', 'epilog', 'formatter_class',
                    'add_help', 'allow_abbrev', 'argument_default'} for name in keywords)):
                continue
            abbreviation = keywords.get('allow_abbrev', ast.Constant(value=True))
            if not isinstance(abbreviation, ast.Constant) or type(abbreviation.value) is not bool:
                continue
            parsers[target] = {'options': {}, 'abbreviate': abbreviation.value}
        elif isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
            parser = parsers.get(function.value.id)
            if parser is None:
                for item in ast.walk(value):
                    if isinstance(item, ast.Name):
                        parsers.pop(item.id, None)
                continue
            if function.attr == 'add_argument':
                if any(not isinstance(arg, ast.Constant) or not isinstance(arg.value, str) for arg in value.args):
                    parsers.pop(function.value.id, None)
                    continue
                flags = tuple(arg.value for arg in value.args if arg.value.startswith('-'))
                required = any(item.arg == 'required' and isinstance(item.value, ast.Constant)
                               and item.value.value is True for item in value.keywords)
                for flag in flags:
                    parser['options'][flag] = (flags, required)
            elif function.attr in {'parse_args', 'parse_known_args'} and not value.args and not value.keywords:
                provided = set()
                for argument in arguments:
                    if argument == '--':
                        break
                    flag = argument.split('=', 1)[0]
                    matches = ([flag] if flag in parser['options'] else
                        [option for option in parser['options'] if parser['abbreviate']
                         and flag.startswith('--') and option.startswith(flag)])
                    if len(matches) == 1:
                        provided.update(parser['options'][matches[0]][0])
                for flags, required in set(parser['options'].values()):
                    if required and not provided.intersection(flags):
                        missing.append('/'.join(flags))
            elif function.attr != 'set_defaults':
                parsers.pop(function.value.id, None)
        else:
            for item in ast.walk(value):
                if isinstance(item, ast.Name):
                    parsers.pop(item.id, None)  # A helper may configure or replace the parser.
    return sorted(set(missing))

"""Find literal output parents in unchanged author Python, without evaluating it."""
import ast


def output_directories(source):
    tree = ast.parse(source)
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.value)
    globals_ = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)
                and len(assignments.get(target.id, [])) == 1}

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + '.' + node.attr
        return ''

    def prefix(node, visited=()):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, True
        if isinstance(node, ast.Name) and node.id in globals_ and node.id not in visited:
            return prefix(globals_[node.id], (*visited, node.id))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, complete = prefix(node.left, visited)
            if not complete:
                return left, False
            right, complete = prefix(node.right, visited)
            return left + right, complete
        if isinstance(node, ast.JoinedStr):
            result = ''
            for part in node.values:
                text, complete = prefix(part, visited)
                result += text
                if not complete:
                    return result, False
            return result, True
        if isinstance(node, ast.Call) and dotted(node.func) == 'os.path.join' and node.args and not node.keywords:
            result = ''
            for part in node.args:
                text, complete = prefix(part, visited)
                result += ('/' if result else '') + text
                if not complete:
                    return result, False
            return result, True
        return '', False

    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name, target = dotted(node.func), None
        if name == 'torch.save' and len(node.args) >= 2:
            target = node.args[1]
        elif name == 'open' and node.args:
            mode = node.args[1] if len(node.args) > 1 else next(
                (item.value for item in node.keywords if item.arg == 'mode'), None)
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(
                    flag in mode.value for flag in 'wax'):
                target = node.args[0]
        if target is None:
            continue
        text, _ = prefix(target)
        text = text.replace('\\', '/')
        # Only an explicit directory separator before the unknown filename can
        # authorize a parent. A dynamic directory or ambiguous name is skipped.
        if '/' not in text:
            continue
        directory = text.rsplit('/', 1)[0]
        if directory and directory != '.':
            result.append({'directory': directory, 'line': node.lineno,
                           'statement': ast.get_source_segment(source, node)})
    return result

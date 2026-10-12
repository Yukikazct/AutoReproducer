"""Read actual GPU placement and original CLI options without importing author code."""
import ast
import os
import shutil
import subprocess


def probe_execution_host():
    executable = shutil.which('nvidia-smi')
    result = {'cpu_available': True, 'cuda_device_indices': [], 'gpu_probe_confirmed': False}
    if not executable:
        return result
    try:
        completed = subprocess.run([executable, '--query-gpu=index', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=10, check=False)
        rows = completed.stdout.strip().splitlines()
        if completed.returncode or not rows or any(not row.strip().isdigit() for row in rows):
            return result
        physical = [int(row.strip()) for row in rows]
        mask = os.environ.get('CUDA_VISIBLE_DEVICES')
        if mask is not None:
            selected = [value.strip() for value in mask.split(',') if value.strip()]
            if any(not value.isdigit() or int(value) not in physical for value in selected):
                return result  # UUID masks need an actual CUDA runtime probe; do not guess.
            physical = list(dict.fromkeys(int(value) for value in selected))
        result.update(cuda_device_indices=list(range(len(physical))), gpu_probe_confirmed=True)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def placement_options(packet, script):
    """Only integer CLI fields used exclusively by torch.cuda.set_device may vary."""
    source = packet['repository']['files'].get(script)
    if not source or not script.endswith('.py'):
        return {}
    tree = ast.parse(source['text'])
    nodes = list(ast.walk(tree))
    if (not any(isinstance(node, ast.Import) and any(alias.name == 'torch' and alias.asname is None for alias in node.names)
                for node in nodes)
            or any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == 'torch' for node in nodes)):
        return {}
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    parsed = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'parse_args'):
            parsed.add(node.targets[0].id)
    bindings = {}
    for node in nodes:
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name) or node.value.id not in parsed:
            continue
        parent = parents.get(node)
        if (not isinstance(parent, ast.Call) or parent.args != [node] or parent.keywords
                or ast.dump(parent.func) != ast.dump(ast.parse('torch.cuda.set_device', mode='eval').body)):
            bindings[(node.value.id, node.attr)] = None
        elif (node.value.id, node.attr) not in bindings:
            bindings[(node.value.id, node.attr)] = ast.get_source_segment(source['text'], parent)
    options = {}
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
            continue
        flags = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith('--')]
        keywords = {item.arg: item.value for item in node.keywords}
        if not flags or ast.dump(keywords.get('type', ast.Constant(None))) != ast.dump(ast.Name('int', ast.Load())):
            continue
        dest = keywords.get('dest', ast.Constant(flags[0][2:].replace('-', '_')))
        if not isinstance(dest, ast.Constant) or not isinstance(dest.value, str):
            continue
        statement = next((bindings[(name, dest.value)] for name in parsed if bindings.get((name, dest.value))), None)
        if statement:
            for flag in flags:
                options[flag] = {'path': script, 'dest': dest.value, 'statement': statement, 'source_sha256': source['sha256']}
    return options

"""Source-bound test-count derivation for the reviewed stratified split recipe."""
import ast
from copy import deepcopy
import math


# Match the entire original algorithms, not isolated numbers or code fragments.
RATE_BODY = '''
dense_split = DENSE
sparse_split = SPARSE
if split == "dense":
    u_split = dense_split
elif split == "sparse":
    u_split = sparse_split
else:
    raise NotImplementedError("split is dense or sparse")
percls_trn = int(round(u_split[0] * len(data.y) / data.num_classes))
val_lb = int(round(u_split[1] * len(data.y)))
train_mask, val_mask, test_mask = du.SAMPLER(data, data.num_classes, percls_trn, val_lb)
dev = data.x.device
mask = torch.empty((data.x.shape[0]), dtype=torch.int8, device=dev)
mask[train_mask] = 0
mask[val_mask] = 1
mask[test_mask] = 2
return mask
'''
SAMPLER_BODY = '''
indices = []
for i in range(num_classes):
    index = (data.y == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0), device=index.device)]
    indices.append(index)
train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
if Flag == 0:
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    train_mask = index_to_mask(train_index, size=data.num_nodes)
    val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
    test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
else:
    val_index = torch.cat([i[percls_trn:percls_trn + val_lb] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn + val_lb:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    train_mask = index_to_mask(train_index, size=data.num_nodes)
    val_mask = index_to_mask(val_index, size=data.num_nodes)
    test_mask = index_to_mask(rest_index, size=data.num_nodes)
return train_mask, val_mask, test_mask
'''
MASK_BODY = '''
mask = torch.zeros(size, dtype=torch.bool, device=index.device)
mask[index] = 1
return mask
'''


def _function(files, location):
    if (not isinstance(location, dict) or set(location) != {'path', 'function'}
            or location.get('path') not in files or not isinstance(location.get('function'), str)):
        raise ValueError('Split count needs a pinned source path and function')
    tree = ast.parse(files[location['path']]['text'])
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == location['function']]
    if (len(functions) != 1 or functions[0] not in tree.body
            or not isinstance(functions[0], ast.FunctionDef) or functions[0].decorator_list):
        raise ValueError('Split count source must be one undecorated top-level function')
    node = deepcopy(functions[0])
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        if isinstance(node.body[0].value.value, str):
            node.body.pop(0)
    return node


def _body_dump(body):
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def _cites_function(reference, path, wanted):
    if reference.get('path') != path:
        return False
    try:
        tree = ast.parse(reference['quote'])
    except (SyntaxError, KeyError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    body = body[1:]
            if _body_dump(body) == wanted:
                return True
    return False


def compile_split_rule(rule, files, *, selected_mode, references):
    fields = {'kind', 'mode', 'rate_source', 'sampler_source'}
    if not isinstance(rule, dict) or set(rule) != fields or rule.get('kind') != 'stratified_remainder':
        raise ValueError('Derived test counts must use the source-bound stratified remainder schema')
    if rule['mode'] not in {'dense', 'sparse'} or selected_mode != rule['mode']:
        raise ValueError('Derived split mode must match the effective original author command')
    rate = _function(files, rule['rate_source'])
    sampler = _function(files, rule['sampler_source'])
    imports = ast.parse(files[rule['rate_source']['path']]['text']).body
    modules = [alias.name for node in imports if isinstance(node, ast.Import)
               for alias in node.names if alias.asname == 'du']
    if modules != [rule['sampler_source']['path'].removesuffix('.py').replace('/', '.')]:
        raise ValueError('Split sampler must be the actual source module imported by the rate function')
    mask = _function(files, {'path': rule['sampler_source']['path'], 'function': 'index_to_mask'})
    if [arg.arg for arg in mask.args.args] != ['index', 'size'] or _body_dump(mask.body) != _body_dump(ast.parse(MASK_BODY).body):
        raise ValueError('Original split mask helper does not preserve the complete author indices')
    for location, node in ((rule['rate_source'], rate), (rule['sampler_source'], sampler)):
        wanted = _body_dump(node.body)
        if not any(_cites_function(ref, location['path'], wanted) for ref in references):
            raise ValueError('Derived counts require citations containing each complete original split function')
    if [arg.arg for arg in rate.args.args] != ['data', 'split'] or rate.args.kwonlyargs or rate.args.vararg or rate.args.kwarg:
        raise ValueError('Unsupported split-rate signature')
    rates = {}
    for name in ('dense_split', 'sparse_split'):
        assignments = [node for node in rate.body if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]
        if len(assignments) != 1:
            raise ValueError('Split rates must be literal original assignments')
        rates[name] = ast.literal_eval(assignments[0].value)
        if (not isinstance(rates[name], list) or len(rates[name]) != 2
                or any(type(value) not in (float, int) or not math.isfinite(value) or not 0 < value < 1
                       for value in rates[name]) or sum(rates[name]) >= 1):
            raise ValueError('Split rates must leave a positive complete test remainder')
    expected = RATE_BODY.replace('DENSE', repr(rates['dense_split'])).replace('SPARSE', repr(rates['sparse_split']))
    expected = expected.replace('SAMPLER', rule['sampler_source']['function'])
    if _body_dump(rate.body) != _body_dump(ast.parse(expected).body):
        raise ValueError('Author split-rate computation differs from the reviewed complete recipe')
    if ([arg.arg for arg in sampler.args.args] != ['data', 'num_classes', 'percls_trn', 'val_lb', 'Flag']
            or sampler.args.kwonlyargs or sampler.args.vararg or sampler.args.kwarg
            or not sampler.args.defaults or ast.literal_eval(sampler.args.defaults[-1]) != 0
            or _body_dump(sampler.body) != _body_dump(ast.parse(SAMPLER_BODY).body)):
        raise ValueError('Author stratified sampler differs from the reviewed complete remainder recipe')
    train, valid = rates[rule['mode'] + '_split']
    return {**deepcopy(rule), 'train_fraction': train, 'validation_fraction': valid}


def expected_from_census(rule, census):
    """Count from all original labels, independently of the selected test mask."""
    if (not isinstance(census, list) or not census or len(census) > 100000
            or any(type(count) is not int or not 0 < count <= 10000000 for count in census)):
        raise ValueError('Derived counts need a complete contiguous-class label census')
    total, classes = sum(census), len(census)
    per_class = round(rule['train_fraction'] * total / classes)
    validation = round(rule['validation_fraction'] * total)
    # Short classes exhaust their actual population, as the author's slices do.
    remaining = total - sum(min(count, per_class) for count in census)
    expected = max(0, remaining - validation)
    if not 0 < expected <= 10000000:
        raise ValueError('The source-derived split has no valid complete test set')
    return expected

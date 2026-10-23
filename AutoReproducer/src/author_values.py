"""Read exact author values through fixed, non-executable selectors."""
import inspect
import re
import sys
import types

_IDENTIFIER = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,127}')


def selector_parts(selector):
    """Validate a fixed selector and return its root plus ordered access steps.

    This schema is shared by the planner, AST comparison and runtime reader so
    an unchecked receipt cannot turn fixed storage access into Python execution.
    Legacy single-kind paths retain their representation in saved plans.
    """
    if isinstance(selector, str):
        name, steps = selector, []
    elif type(selector) is dict and set(selector) in (
            {'name', 'attributes'}, {'name', 'keys'}, {'name', 'path'}):
        name = selector['name']
        field = next(key for key in ('attributes', 'keys', 'path') if key in selector)
        values = selector[field]
        if type(values) is not list or not 1 <= len(values) <= 8:
            raise ValueError('Author selectors need one to eight fixed storage access steps')
        if field == 'attributes':
            steps = [('attribute', value) for value in values]
        elif field == 'keys':
            steps = [('key', value) for value in values]
        else:
            if any(type(step) is not dict or set(step) not in ({'attribute'}, {'key'}) for step in values):
                raise ValueError('Author selector path steps must contain exactly one attribute or key')
            steps = [(kind, value) for step in values for kind, value in step.items()]
    else:
        raise ValueError('Author selectors must be identifiers or fixed key/stored-field paths')
    if type(name) is not str or not _IDENTIFIER.fullmatch(name):
        raise ValueError('Author selector root must be an original public identifier; mixed attribute/key '
                         'access requires a JSON object with name and ordered path, never an expression string')
    for kind, value in steps:
        if kind == 'attribute':
            if type(value) is not str or not _IDENTIFIER.fullmatch(value):
                raise ValueError('Author stored-field selector must name a public fixed field')
        elif not (type(value) is str and 1 <= len(value) <= 128
                  or type(value) is int and 0 <= value < 10000000):
            raise ValueError('Author key selector must use a bounded fixed string or nonnegative integer')
    return name, steps


def stored_fields(value):
    descriptor = inspect.getattr_static(type(value), '__dict__', None)
    if not isinstance(descriptor, types.GetSetDescriptorType):
        raise ValueError('Author stored-field selector requires an ordinary instance dictionary')
    stored = descriptor.__get__(value, type(value))
    if type(stored) is not dict:
        raise ValueError('Author stored-field selector did not resolve actual instance storage')
    return stored


def geometric_data_fields(value):
    """Read only the reviewed, exact PyG Data/GlobalStorage classes' raw mapping."""
    module = sys.modules.get('torch_geometric.data.data')
    storage = sys.modules.get('torch_geometric.data.storage')
    if type(module) is not types.ModuleType or type(storage) is not types.ModuleType:
        return None
    if type(value) is not module.__dict__.get('Data'):
        return None
    backing = stored_fields(value).get('_store')
    if type(backing) is not storage.__dict__.get('GlobalStorage'):
        raise ValueError('Author PyG Data does not use the reviewed original GlobalStorage')
    mapping = stored_fields(backing).get('_mapping')
    if type(mapping) is not dict:
        raise ValueError('Author PyG Data does not expose its complete directly stored field mapping')
    return mapping


def resolve_author_value(namespace, selector):
    name, steps = selector_parts(selector)
    value = namespace[name]
    for kind, key in steps:
        if kind == 'attribute':
            # Use the built-in storage descriptor, never a property or getter.
            stored = geometric_data_fields(value)
            if stored is None:
                stored = stored_fields(value)
            if type(stored) is not dict or key not in stored:
                raise ValueError('Author field is not directly stored; properties and dynamic attributes are unsupported')
            value = stored[key]
        else:
            if type(value) not in (dict, list, tuple):
                raise ValueError('Author key selector must traverse actual dictionary/list values')
            if type(value) in (list, tuple) and type(key) is not int:
                raise ValueError('Author sequence key must be a fixed nonnegative integer')
            value = value[key]
    return value

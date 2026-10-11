"""Read exact author values through fixed, non-executable selectors."""
import inspect
import sys
import types


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
    if isinstance(selector, str):
        return namespace[selector]
    value = namespace[selector['name']]
    if 'attributes' in selector:
        for name in selector['attributes']:
            # Use the built-in storage descriptor, never a property or getter.
            stored = geometric_data_fields(value)
            if stored is None:
                stored = stored_fields(value)
            if type(stored) is not dict or name not in stored:
                raise ValueError('Author field is not directly stored; properties and dynamic attributes are unsupported')
            value = stored[name]
        return value
    for key in selector['keys']:
        if type(value) not in (dict, list, tuple):
            raise ValueError('Author key selector must traverse actual dictionary/list values')
        value = value[key]
    return value

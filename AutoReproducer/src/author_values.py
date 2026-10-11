"""Read exact author values through fixed, non-executable selectors."""
import inspect
import types


def resolve_author_value(namespace, selector):
    if isinstance(selector, str):
        return namespace[selector]
    value = namespace[selector['name']]
    if 'attributes' in selector:
        for name in selector['attributes']:
            # Use the built-in storage descriptor, never a property or getter.
            descriptor = inspect.getattr_static(type(value), '__dict__', None)
            if not isinstance(descriptor, types.GetSetDescriptorType):
                raise ValueError('Author stored-field selector requires an ordinary instance dictionary')
            stored = descriptor.__get__(value, type(value))
            if type(stored) is not dict or name not in stored:
                raise ValueError('Author field is not directly stored; properties and dynamic attributes are unsupported')
            value = stored[name]
        return value
    for key in selector['keys']:
        if type(value) not in (dict, list, tuple):
            raise ValueError('Author key selector must traverse actual dictionary/list values')
        value = value[key]
    return value

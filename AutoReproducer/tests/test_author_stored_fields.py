"""Selectors read stored author fields without executing properties or getters."""
import pytest

from src.author_values import resolve_author_value, selector_parts
from src.discovered_repository_runtime import _selector_names


def test_stored_fields_bypass_custom_getattribute_and_keep_exact_value():
    marker = object()
    class AuthorRecord:
        def __init__(self):
            self.inputs = marker
        def __getattribute__(self, name):
            raise AssertionError('An attribute getter must not execute during capture')
    assert resolve_author_value({'record': AuthorRecord()},
        {'name': 'record', 'attributes': ['inputs']}) is marker


@pytest.mark.parametrize('kind', ['property', 'getattr', 'dictionary_property'])
def test_computed_fields_are_rejected_without_calling_author_code(kind):
    class AuthorRecord:
        @property
        def inputs(self):
            raise AssertionError('Property must never execute')
        def __getattr__(self, name):
            raise AssertionError('Dynamic getter must never execute')
    if kind == 'dictionary_property':
        class DictionaryProperty(AuthorRecord):
            @property
            def __dict__(self):
                raise AssertionError('Dictionary getter must never execute')
        AuthorRecord = DictionaryProperty
    with pytest.raises((ValueError, AttributeError, TypeError)):
        resolve_author_value({'record': AuthorRecord()}, {'name': 'record', 'attributes': ['inputs']})


@pytest.mark.parametrize('fields', [['inputs()'], ['__class__'], [], ['x'] * 9, [1]])
def test_stored_field_schema_rejects_expressions_private_fields_and_unbounded_paths(fields):
    with pytest.raises(ValueError):
        _selector_names({'name': 'record', 'attributes': fields})


def test_mixed_paths_preserve_step_order_exact_values_and_legacy_formats():
    marker = object()
    class AuthorRecord:
        def __init__(self):
            self.graph = {'inputs': [{'item': marker}]}
        def __getattribute__(self, name):
            raise AssertionError('Capture must read raw storage without a getter')
    selector = {'name': 'record', 'path': [
        {'attribute': 'graph'}, {'key': 'inputs'}, {'key': 0}, {'key': 'item'}]}
    assert _selector_names(selector) == ['record']
    assert resolve_author_value({'record': AuthorRecord()}, selector) is marker
    assert resolve_author_value({'records': [AuthorRecord()]}, {'name': 'records', 'path': [
        {'key': 0}, {'attribute': 'graph'}, {'key': 'inputs'}, {'key': 0}, {'key': 'item'}]}) is marker
    assert resolve_author_value({'record': marker}, 'record') is marker
    assert resolve_author_value({'records': {'inputs': [marker]}},
        {'name': 'records', 'keys': ['inputs', 0]}) is marker


@pytest.mark.parametrize('path', [
    [], [{'attribute': 'x'}] * 9, [{'attribute': '__dict__'}], [{'attribute': 'x()'}],
    [{'attribute': 'x', 'key': 'y'}], [{'call': 'x'}], ['x'], [{'key': True}],
    [{'key': -1}], [{'key': 10000000}], [{'key': slice(0, 1)}], [{'key': ''}],
    [{'key': 'x' * 129}], [{'attribute': 1}], [{'key': {'name': 'dynamic'}}],
])
def test_mixed_path_schema_rejects_executable_ambiguous_or_unbounded_steps(path):
    selector = {'name': 'record', 'path': path}
    with pytest.raises(ValueError):
        selector_parts(selector)
    with pytest.raises(ValueError):
        resolve_author_value({'record': object()}, selector)


@pytest.mark.parametrize('kind', ['property', 'getattr', 'dict_subclass', 'custom_index'])
def test_mixed_paths_never_execute_computed_fields_or_custom_indexing(kind):
    class AuthorRecord:
        @property
        def graph(self):
            raise AssertionError('Property must never execute')
        def __getattr__(self, name):
            raise AssertionError('Dynamic getter must never execute')
    class CustomIndex:
        def __getitem__(self, key):
            raise AssertionError('Custom indexing must never execute')
    class Dictionary(dict):
        def __getitem__(self, key):
            raise AssertionError('Dictionary subclass indexing must never execute')
    record = AuthorRecord()
    if kind in ('dict_subclass', 'custom_index'):
        object.__getattribute__(record, '__dict__')['graph'] = (
            Dictionary(inputs=object()) if kind == 'dict_subclass' else CustomIndex())
    path = [{'attribute': 'missing' if kind == 'getattr' else 'graph'}, {'key': 'inputs'}]
    with pytest.raises(ValueError):
        resolve_author_value({'record': record}, {'name': 'record', 'path': path})

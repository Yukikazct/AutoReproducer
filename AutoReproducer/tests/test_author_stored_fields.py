"""Selectors read stored author fields without executing properties or getters."""
import pytest

from src.author_values import resolve_author_value
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

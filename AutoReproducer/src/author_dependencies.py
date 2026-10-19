"""Shared distribution identity for declared and runtime author dependencies."""
import re

from packaging.requirements import InvalidRequirement, Requirement

from src.agents.dependency_resolver import python_package_for


def normalize_package(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def unformat_requirement(line):
    line = re.sub(r'^\s*(?:[-*+]|\d+[.)])\s+', '', line).strip()
    return line[1:-1] if len(line) >= 2 and line.startswith('`') and line.endswith('`') else line


def import_distribution(module):
    return normalize_package(python_package_for(module))


def declared_requirements(module, author):
    """Keep markers, extras and URLs; callers must not silently discard them."""
    package = import_distribution(module)
    found = {}
    for line in author:
        try:
            requirement = Requirement(unformat_requirement(line))
        except InvalidRequirement:
            continue
        if normalize_package(requirement.name) == package:
            found[str(requirement)] = requirement
    return list(found.values())


def has_declaration(value, author):
    expected = Requirement(value)
    for line in author:
        try:
            if Requirement(unformat_requirement(line)) == expected:
                return True
        except InvalidRequirement:
            continue
    return False

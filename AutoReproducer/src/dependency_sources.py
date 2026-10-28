"""Keep reviewed native wheels from becoming an index for ordinary packages."""
from __future__ import annotations

import re

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


_NATIVE_PACKAGES = ("torch", "torchvision", "torchaudio")
_PYG_NATIVE_PACKAGES = {"torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "pyg-lib"}
_EXTRA_INDEX = re.compile(
    r"--extra-index-url(?:\s+|=)https://download\.pytorch\.org/whl/(cpu|cu121)/?"
)
_REVIEWED_LINKS = re.compile(
    r"--find-links(?:\s+|=)https://data\.pyg\.org/whl/torch-2\.5\.0\+(cpu|cu121)\.html"
)


def scope_pytorch_sources(requirements: str, *, default_cpu_links: bool = False) -> str:
    """Return installer-only text with package-scoped official wheel pages.

    The caller must first establish that this is an automatically prepared,
    reviewed public environment. Its original requirements remain the source
    evidence. A changed return value also allows the caller to omit its global
    default CPU find-links page; explicit custom links must remain unchanged.

    pip gives extra indexes no lower priority than the primary index. Scoping
    the reviewed PyTorch source to its explicitly pinned native packages keeps
    ordinary dependencies on the selected primary mirror. Ambiguous inputs
    retain their original routing and text exactly.
    """
    lines = requirements.splitlines(keepends=True)
    supplemental = None
    native = {}
    pyg_native = {}
    pyg_flavors = set()
    for index, raw in enumerate(lines):
        # Match pip's requirements-file comment rule; preserve the actual line.
        line = re.sub(r"(^|\s+)#.*$", "", raw.strip()).strip()
        if not line:
            continue
        if line.endswith("\\"):
            return requirements
        source = _EXTRA_INDEX.fullmatch(line)
        if source:
            if supplemental is not None:
                return requirements
            supplemental = (index, source[1])
            continue
        if line.startswith("-"):
            pyg_source = _REVIEWED_LINKS.fullmatch(line)
            if not pyg_source:
                return requirements
            pyg_flavors.add(pyg_source[1])
            continue
        try:
            item = Requirement(line)
        except InvalidRequirement:
            return requirements
        if item.url:
            return requirements
        name = canonicalize_name(item.name)
        if name == "torch-geometric" and not item.extras:
            continue  # Its ordinary Python distribution comes from PyPI.
        if name not in _NATIVE_PACKAGES and name not in _PYG_NATIVE_PACKAGES:
            # Such packages can require additional wheels available only from
            # the broad PyTorch index; their routing has not been reviewed.
            if name.startswith(("torch", "pytorch-")) or name == "triton":
                return requirements
            continue
        pins = list(item.specifier)
        target = pyg_native if name in _PYG_NATIVE_PACKAGES else native
        if (name in target or item.extras or len(pins) != 1
                or pins[0].operator != "==" or "*" in pins[0].version):
            return requirements
        try:
            version = Version(pins[0].version)
        except InvalidVersion:
            return requirements
        target[name] = (version, item.marker)

    if "torch" not in native or (supplemental is None and not default_cpu_links):
        return requirements
    flavor = supplemental[1] if supplemental else "cpu"
    if pyg_flavors and pyg_flavors != {flavor}:
        return requirements
    if pyg_native and (pyg_flavors != {flavor}
                       or native["torch"][0].release[:2] != (2, 5)
                       or any(version.local != "pt25" + flavor for version, _ in pyg_native.values())):
        return requirements
    for version, marker in native.values():
        if version.local is not None and version.local != flavor:
            return requirements
        # A conditionally absent torch pin cannot constrain an unconditional
        # torchvision/torchaudio dependency on an unknown torch build.
        if native["torch"][1] is not None and marker != native["torch"][1]:
            return requirements
    links = [f"--find-links https://download.pytorch.org/whl/{flavor}/{name}/"
             for name in _NATIVE_PACKAGES if name in native]
    newline = "\r\n" if "\r\n" in requirements else "\n"
    if supplemental:
        index = supplemental[0]
        raw = lines[index]
        ending = raw[len(raw.rstrip("\r\n")):]
        indentation = raw[:len(raw) - len(raw.lstrip())]
        comment = re.search(r"\s+#.*", raw.rstrip("\r\n"))
        replacement = [indentation + line for line in links]
        if comment:
            replacement[0] += comment[0]
        lines[index] = newline.join(replacement) + ending
        return "".join(lines)
    separator = "" if not requirements or requirements.endswith(("\r", "\n")) else newline
    return requirements + separator + newline.join(links) + newline

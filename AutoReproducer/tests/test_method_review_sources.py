"""Online method review uses complete, separately attributed pinned sources."""
from copy import deepcopy
import hashlib

import pytest

from src.method_adapters import NeuralODEAdapter
from src.method_profiles import ODE_SHA, method_profile


OPTIONAL_AUTHOR_FILES = {
    "torchdiffeq/_impl/odeint.py": (
        "def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):\n"
        "    return None\n" + "# full solver API documentation\n" * 200 + "# end solver API\n"),
    "torchdiffeq/_impl/misc.py": (
        "def _check_inputs(method):\n"
        "    if method is None:\n        method = 'dopri5'\n"
        "    return method\n" + "# full solver input validation\n" * 200 + "# end solver defaults\n"),
    "setup.py": (
        "import setuptools\nsetuptools.setup(\n"
        "    install_requires=['torch>=1.5.0', 'scipy>=1.4.0'],\n"
        "    python_requires='~=3.6',\n)\n"
        + "# full package metadata\n" * 200 + "# end package requirements\n"),
}


@pytest.fixture
def official_workspace(tmp_path):
    contents = {
        "examples/ode_demo.py": (
            "# official example start\n"
            "class ODEFunc:\n    def forward(self, t, y):\n        return self.net(y**3)\n"
            + "# preserved source line\n" * 240
            + "# official training and evaluation end\n"),
        "README.md": "# Differentiable ODE solvers\n" + "Full author documentation.\n" * 300
                     + "Neural Ordinary Differential Equations, 2018.\n",
        "torchdiffeq/__init__.py": "from ._impl import odeint\nfrom ._impl import odeint_adjoint\n__version__ = '0.2.5'\n",
    }
    for name, content in contents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path, contents


def test_neural_ode_review_sources_are_complete_and_revision_pinned(official_workspace):
    workspace, contents = official_workspace
    profile = method_profile("neural_ode_spiral")
    sources = NeuralODEAdapter().public_sources(workspace, profile)
    assert [source["source_id"] for source in sources] == [
        "author_ode_demo", "author_ode_readme", "author_ode_package_init"]
    assert {source["locator"] for source in sources} == set(profile["required_files"])
    for source in sources:
        assert source["content"] == contents[source["locator"]]
        assert source["url"] == f"https://github.com/rtqichen/torchdiffeq/blob/{ODE_SHA}/{source['locator']}"
        assert source["origin"] == "official_repository"
        assert source["revision"] == ODE_SHA
        assert source["sha256"] == hashlib.sha256(source["content"].encode("utf-8")).hexdigest()
    assert sources[0]["url"] == profile["paper"]["reference_source"]
    assert sources[0]["content"].endswith("# official training and evaluation end\n")
    assert sources[1]["content"].endswith("Neural Ordinary Differential Equations, 2018.\n")


@pytest.mark.parametrize("required", ["examples/ode_demo.py", "README.md", "torchdiffeq/__init__.py"])
def test_review_sources_do_not_silently_omit_missing_pinned_documentation(official_workspace, required):
    workspace, _ = official_workspace
    (workspace / required).unlink()
    with pytest.raises(FileNotFoundError):
        NeuralODEAdapter().public_sources(workspace, method_profile("neural_ode_spiral"))


def test_optional_author_sources_support_library_defaults_and_dependency_bounds(official_workspace):
    workspace, required_contents = official_workspace
    for locator, content in OPTIONAL_AUTHOR_FILES.items():
        path = workspace / locator
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    profile = method_profile("neural_ode_spiral")
    frozen_profile = deepcopy(profile)

    sources = NeuralODEAdapter().public_sources(workspace, profile)

    assert [source["source_id"] for source in sources] == [
        "author_ode_demo", "author_ode_readme", "author_ode_package_init",
        "author_ode_solver_api", "author_ode_solver_defaults", "author_ode_package_requirements"]
    expected_contents = {**required_contents, **OPTIONAL_AUTHOR_FILES}
    for source in sources:
        assert source["content"] == expected_contents[source["locator"]]
        assert source["url"] == f"{profile['repository']['url']}/blob/{ODE_SHA}/{source['locator']}"
        assert source["origin"] == "official_repository" and source["revision"] == ODE_SHA
        assert source["sha256"] == hashlib.sha256(source["content"].encode("utf-8")).hexdigest()
    evidence = {source["source_id"]: source["content"] for source in sources}
    assert "rtol=1e-7, atol=1e-9, method=None" in evidence["author_ode_solver_api"]
    assert "if method is None:\n        method = 'dopri5'" in evidence["author_ode_solver_defaults"]
    assert "install_requires=['torch>=1.5.0', 'scipy>=1.4.0']" in evidence["author_ode_package_requirements"]
    assert "python_requires='~=3.6'" in evidence["author_ode_package_requirements"]
    assert "torch==2.5.1+cpu" not in evidence["author_ode_package_requirements"]
    assert profile == frozen_profile


@pytest.mark.parametrize("locator", list(OPTIONAL_AUTHOR_FILES))
def test_optional_author_sources_can_be_present_individually(official_workspace, locator):
    workspace, _ = official_workspace
    path = workspace / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OPTIONAL_AUTHOR_FILES[locator], encoding="utf-8")

    sources = NeuralODEAdapter().public_sources(workspace, method_profile("neural_ode_spiral"))

    assert len(sources) == 4
    assert [source["locator"] for source in sources[:3]] == [
        "examples/ode_demo.py", "README.md", "torchdiffeq/__init__.py"]
    assert sources[-1]["locator"] == locator
    assert sources[-1]["content"] == OPTIONAL_AUTHOR_FILES[locator]


def test_demo_materialization_still_requires_its_frozen_content_hash(official_workspace, monkeypatch):
    workspace, contents = official_workspace
    profile = method_profile("neural_ode_spiral")
    adapter = NeuralODEAdapter()
    with pytest.raises(ValueError, match="源码校验失败"):
        adapter.materialize(workspace, profile, "spec")
    profile["source_sha256"] = hashlib.sha256(contents["examples/ode_demo.py"].encode()).hexdigest()
    monkeypatch.setattr(adapter, "_write_runtime", lambda *args: {"verified": True})
    assert adapter.materialize(workspace, profile, "spec") == {"verified": True}
    assert "class ODEFunc" in (workspace / "author_model.py").read_text(encoding="utf-8")
    # Reviewing additional documents never replaces the original demo evidence.
    assert adapter.public_sources(workspace, profile)[0]["content"] == contents["examples/ode_demo.py"]

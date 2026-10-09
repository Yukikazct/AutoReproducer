"""Online method review uses complete, separately attributed pinned sources."""
import hashlib

import pytest

from src.method_adapters import NeuralODEAdapter
from src.method_profiles import ODE_SHA, method_profile


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
    assert sources[0]["url"] == profile["paper"]["reference_source"]
    assert sources[0]["content"].endswith("# official training and evaluation end\n")
    assert sources[1]["content"].endswith("Neural Ordinary Differential Equations, 2018.\n")


def test_review_sources_do_not_silently_omit_missing_pinned_documentation(official_workspace):
    workspace, _ = official_workspace
    (workspace / "README.md").unlink()
    with pytest.raises(FileNotFoundError):
        NeuralODEAdapter().public_sources(workspace, method_profile("neural_ode_spiral"))


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

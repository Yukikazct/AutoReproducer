"""Load the project's ignored local API settings for both page and CLI."""
import os
from pathlib import Path
import tomllib


def load_local_llm_settings(project_root=None):
    root = Path(project_root or Path(__file__).resolve().parents[1])
    path = root / ".streamlit" / "secrets.toml"
    if not path.is_file():
        return
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        value = settings.get(name)
        if isinstance(value, str) and value.strip() and not os.environ.get(name):
            os.environ[name] = value.strip()

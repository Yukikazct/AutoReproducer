"""Persist sandbox figures and configure headless, Chinese-capable plotting."""
import base64
import hashlib
import os
import re
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "reports" / "artifacts"
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "NotoSansSC.ttf"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_IMAGES = 32
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}


def prepare_plot_runtime(workdir: str, docker: bool = False) -> dict:
    """Keep generated code unchanged; initialize Matplotlib in the subprocess."""
    runtime = Path(workdir) / ".autorepro_plot"
    runtime.mkdir(exist_ok=True)
    (runtime / "matplotlibrc").write_text(
        "backend: Agg\nfont.family: sans-serif\n"
        "font.sans-serif: Noto Sans SC, DejaVu Sans\n"
        "axes.unicode_minus: False\n", encoding="utf-8")
    # sitecustomize is loaded from PYTHONPATH at Python startup. Missing
    # Matplotlib remains the executor's normal dependency/self-heal concern.
    (runtime / "sitecustomize.py").write_text(
        "import os\n"
        "try:\n"
        "    from matplotlib import font_manager, rcParams\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    font = os.environ.get('AUTOREPRO_FONT_PATH', '')\n"
        "    if os.path.isfile(font):\n"
        "        font_manager.fontManager.addfont(font)\n"
        "        name = font_manager.FontProperties(fname=font).get_name()\n"
        "        rcParams['font.sans-serif'] = [name, 'DejaVu Sans']\n",
        encoding="utf-8")
    if docker:
        return {"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/autorepro-matplotlib",
                "MATPLOTLIBRC": "/app/.autorepro_plot/matplotlibrc",
                "AUTOREPRO_FONT_PATH": "/autorepro-fonts/NotoSansSC.ttf",
                "PYTHONPATH": "/app/.autorepro_plot"}
    cache = runtime / "cache"
    cache.mkdir(exist_ok=True)
    return {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(cache.resolve()),
            "MATPLOTLIBRC": str((runtime / "matplotlibrc").resolve()),
            "AUTOREPRO_FONT_PATH": str(FONT_PATH),
            "PYTHONPATH": str(runtime.resolve())}


def collect_images(workdir: str, stage: str, session_id: str = "") -> dict:
    """Copy full-run raster images before cleanup; never follow sandbox links."""
    if stage != "full":
        return {"artifacts": [], "artifact_warnings": []}
    from PIL import Image

    root = Path(workdir).resolve()
    artifacts, warnings = [], []
    dest = None
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            mime = IMAGE_TYPES.get(path.suffix.lower())
            if not mime or path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            try:
                size = path.stat().st_size
                if (size > MAX_IMAGE_BYTES or total + size > MAX_TOTAL_BYTES
                        or len(artifacts) >= MAX_IMAGES):
                    warnings.append(f"图片超出报告收集限额，未嵌入: {relative}")
                    continue
                with Image.open(path) as im:
                    im.verify()
                raw = path.read_bytes()
                if dest is None:
                    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
                    sid = session_id if isinstance(session_id, str) else ""
                    sid = re.sub(r"[^a-zA-Z0-9_-]", "", sid)
                    prefix = f"figures_{sid}_" if sid else "figures_"
                    dest = Path(tempfile.mkdtemp(prefix=prefix, dir=ARTIFACT_ROOT))
                target = dest / f"{len(artifacts) + 1:02d}{path.suffix.lower()}"
                target.write_bytes(raw)
                artifacts.append({"name": relative, "path": str(target),
                                  "mime_type": mime, "bytes": len(raw),
                                  "sha256": hashlib.sha256(raw).hexdigest()})
                total += len(raw)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                warnings.append(f"图片无法收集: {relative} ({exc})")
    return {"artifacts": artifacts, "artifact_warnings": warnings}


def image_data_url(artifact: dict) -> str:
    """Embed only collected, unchanged images; the Markdown stays self-contained."""
    try:
        path = Path(artifact["path"])
        if path.is_symlink() or not path.resolve().is_relative_to(ARTIFACT_ROOT.resolve()):
            return ""
        if path.stat().st_size > MAX_IMAGE_BYTES:
            return ""
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != artifact.get("sha256"):
            return ""
        mime = IMAGE_TYPES.get(path.suffix.lower())
        if not mime:
            return ""
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    except (OSError, KeyError, TypeError, ValueError):
        return ""

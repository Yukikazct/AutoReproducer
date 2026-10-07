"""Persist sandbox figures and configure headless, Chinese-capable plotting."""
import base64
import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote

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
        (PROJECT_ROOT / "src" / "plot_runtime.py").read_text(encoding="utf-8"),
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


def _verified_file_bytes(path: Path, root: Path, sha256: str) -> bytes | None:
    """Read a bounded, unchanged regular file inside its declared asset root."""
    try:
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())
                or not isinstance(sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha256)
                or path.suffix.lower() not in IMAGE_TYPES
                or path.stat().st_size > MAX_IMAGE_BYTES):
            return None
        raw = path.read_bytes()
        if len(raw) > MAX_IMAGE_BYTES or hashlib.sha256(raw).hexdigest() != sha256:
            return None
        return raw
    except (OSError, TypeError, ValueError):
        return None


def verified_image_bytes(artifact: dict) -> bytes | None:
    """Read only original collected images whose recorded SHA-256 still matches."""
    try:
        return _verified_file_bytes(Path(artifact["path"]), ARTIFACT_ROOT,
                                    artifact.get("sha256"))
    except (KeyError, TypeError, ValueError):
        return None


def default_report_path() -> Path:
    """The default report base is stable even when the process changes CWD."""
    return PROJECT_ROOT / "data" / "reports" / "report.md"


def _report_file(report_path: str | Path) -> Path:
    path = Path(report_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.parent.resolve() / path.name


def _report_asset_directory(report_file: Path) -> Path:
    return report_file.parent / f"{report_file.stem}_assets"


def markdown_image_target(artifact: dict, report_path: str | Path) -> str:
    """Copy a verified image beside a report and return its relative URL.

    Every report owns a companion ``<report_stem>_assets`` directory. Moving
    that directory together with the Markdown preserves its image references;
    copying a report to another output location requires calling this helper
    with that new report path. Relative report paths are rooted at PROJECT_ROOT,
    never the process CWD. The original collected image is checked on every call.
    """
    raw = verified_image_bytes(artifact)
    if raw is None:
        return ""
    temporary = None
    try:
        report_file = _report_file(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        directory = _report_asset_directory(report_file)
        if directory.is_symlink():
            return ""
        directory.mkdir(exist_ok=True)
        if directory.resolve().parent != report_file.parent:
            return ""
        suffix = Path(artifact["path"]).suffix.lower()
        target = directory / f"{artifact['sha256']}{suffix}"
        if target.is_symlink():
            return ""
        if _verified_file_bytes(target, directory, artifact["sha256"]) != raw:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
            os.replace(temporary, target)
            temporary = None
        relative = target.relative_to(report_file.parent).as_posix()
        return quote(relative, safe="/")
    except (OSError, KeyError, TypeError, ValueError):
        return ""
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def verified_report_image_bytes(relative_target: str, report_path: str | Path) -> bytes | None:
    """Read a report companion image for preview/download without escaping it.

    The target must be an emitted relative URL, exactly one asset filename
    below this report's companion directory. Its filename supplies the expected
    SHA-256, so changed assets are rejected even after the source run is cleaned.
    """
    try:
        if not isinstance(relative_target, str):
            return None
        relative = Path(unquote(relative_target))
        report_file = _report_file(report_path)
        directory = _report_asset_directory(report_file)
        if (relative.is_absolute() or len(relative.parts) != 2
                or relative.parts[0] != directory.name or directory.is_symlink()):
            return None
        name = relative.parts[1]
        suffix = Path(name).suffix.lower()
        digest = Path(name).stem
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or suffix not in IMAGE_TYPES:
            return None
        return _verified_file_bytes(directory / name, directory, digest)
    except (OSError, TypeError, ValueError):
        return None


def image_data_url(artifact: dict) -> str:
    """Compatibility helper for clients that explicitly support data URLs."""
    raw = verified_image_bytes(artifact)
    if raw is None:
        return ""
    mime = IMAGE_TYPES[Path(artifact["path"]).suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

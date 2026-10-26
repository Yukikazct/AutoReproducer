"""Validate/freeze recovery preregistration or aggregate trusted trial receipts."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.recovery import aggregate_results, freeze_manifest, run_benchmark, validate_manifest
from src.benchmark.recovery_admission import collect_author_source


def _read(path):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Input JSON exceeds 32 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def _callback(spec):
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", spec):
        raise ValueError("Trusted callback must be module:function")
    module, function = spec.split(":", 1)
    callback = getattr(importlib.import_module(module), function)
    if not callable(callback):
        raise ValueError("Trusted callback is not callable")
    return callback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "freeze", "aggregate", "run"):
        child = sub.add_parser(command)
        child.add_argument("--manifest", required=True)
        if command in {"freeze", "run"}:
            child.add_argument("--admission-audit", help="Trusted bounded exposure-review JSON; required for formal completion")
            child.add_argument("--registry", help="Shared acceptance registry; defaults to this application's registry")
        if command in {"freeze", "aggregate"}:
            child.add_argument("--output", required=True, help="New JSON file; existing files are never overwritten")
        if command == "aggregate":
            child.add_argument("--results", required=True, help="JSON list or JSONL of trusted trial receipts")
        if command == "run":
            child.add_argument("--source-roots", required=True, help="JSON object mapping case ids to prepared source directories")
            child.add_argument("--product-roots", required=True, help="JSON object mapping A/B/C to frozen application checkouts")
            child.add_argument("--adapter", required=True, help="Trusted Python module:function trial adapter")
            child.add_argument("--oracle", required=True, help="Trusted Python module:function independent evaluator")
            child.add_argument("--output-dir", required=True, help="New trial output directory outside all source/product roots")
            child.add_argument("--split", choices=("dev", "holdout"), default="holdout")
    source = sub.add_parser("register-source", help="Read a complete author Git checkout into a trusted acquisition receipt")
    source.add_argument("--source-root", required=True)
    source.add_argument("--output", required=True, help="New acquisition JSON file; never overwritten")
    args = parser.parse_args(argv)
    try:
        manifest = None if args.command == "register-source" else _read(args.manifest)
        if args.command == "register-source":
            result = collect_author_source(args.source_root)
        elif args.command == "validate":
            validated = validate_manifest(manifest)
            result = {"valid": True, "frozen": "manifest_sha256" in validated,
                      "cases": len(validated["cases"]), "acceptance_executed": False}
        elif args.command == "freeze":
            result = freeze_manifest(manifest, admission_audit_path=args.admission_audit,
                                     registry_path=args.registry)
        elif args.command == "run":
            result = run_benchmark(manifest, _read(args.source_roots), args.output_dir,
                trial_runner=_callback(args.adapter), oracle_evaluator=_callback(args.oracle),
                product_roots=_read(args.product_roots), split=args.split,
                admission_audit_path=args.admission_audit, registry_path=args.registry)
        else:
            path = Path(args.results)
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("Trial receipts exceed 32 MiB")
            records = ([json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
                       if path.suffix.lower() == ".jsonl" else _read(path))
            result = aggregate_results(manifest, records)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.command in {"freeze", "aggregate", "register-source"}:
            with Path(args.output).open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
        if args.command == "register-source":
            print(json.dumps({"output": str(Path(args.output).resolve()),
                "repository_url": result["repository_url"], "repository_revision": result["repository_revision"],
                "git_tree": result["git_tree"], "file_count": len(result["source_files"]),
                "author_executed": False}, ensure_ascii=False))
        else:
            print(text)
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, AttributeError) as exc:
        print("Recovery evaluation failed: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

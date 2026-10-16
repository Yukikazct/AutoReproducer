"""Validate a complete source patch candidate before touching an isolated workspace."""
from __future__ import annotations

import difflib
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile

from src.safety.patch_policy import MAX_PATCH_FILES_PER_CANDIDATE, PatchPolicy
from src.safety.paths import workspace_path
from src.safety.workspace_snapshot import restore_snapshot, snapshot_workspace


class PatchTransaction:
    """All-or-nothing candidate with rollback unless the caller explicitly commits.

    The caller owns the isolated workspace and must stop its processes before
    entering or leaving the transaction. Shared repository caches are unsuitable.
    Only existing regular source files can be replaced; additions/deletions and
    executable commands are deliberately outside the proposal format.
    """

    def __init__(self, workspace, patches, policy: PatchPolicy):
        self.root = workspace_path(workspace, ".", "patch workspace", must_exist=True)
        self.policy = policy
        self.files = []
        self.snapshot = None
        self.committed = False
        self.restored = []
        self.removed = []
        self.rollback_error = None
        if not isinstance(patches, list) or not 1 <= len(patches) <= MAX_PATCH_FILES_PER_CANDIDATE:
            raise ValueError(f"a candidate requires 1-{MAX_PATCH_FILES_PER_CANDIDATE} patches")
        seen = set()
        for patch in patches:
            if not isinstance(patch, dict):
                raise ValueError("each patch must be an object")
            decision = policy.decide(patch.get("path"), workspace=str(self.root))
            if not decision.allowed:
                raise ValueError(decision.reason)
            path = workspace_path(self.root, decision.path, "patch target", must_exist=True, forbid_git=True)
            key = path.relative_to(self.root).as_posix().casefold()
            if key in seen:
                raise ValueError(f"duplicate patch target: {decision.path}")
            seen.add(key)
            if not path.is_file():
                raise ValueError(f"patch target is not a regular file: {decision.path}")
            content = patch.get("content")
            base = patch.get("base_sha256")
            if not isinstance(content, str) or not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{64}", base):
                raise ValueError("patch requires text content and an exact base_sha256")
            raw = content.encode("utf-8")
            original = path.read_bytes()
            if len(raw) > policy.max_patchable_file_bytes or len(original) > policy.max_patchable_file_bytes:
                raise ValueError(f"patch exceeds the file size budget: {decision.path}")
            if hashlib.sha256(original).hexdigest() != base:
                raise ValueError(f"base_sha256 mismatch: {decision.path}")
            if raw == original:
                raise ValueError(f"patch makes no change: {decision.path}")
            if path.suffix.lower() == ".py":
                compile(raw, decision.path, "exec")
            # Binary/undecodable files cannot enter the text patch protocol.
            before = original.decode("utf-8")
            diff = "".join(difflib.unified_diff(before.splitlines(keepends=True),
                content.splitlines(keepends=True), fromfile="a/" + decision.path,
                tofile="b/" + decision.path))
            self.files.append({"path": decision.path, "base_sha256": base,
                "sha256": hashlib.sha256(raw).hexdigest(), "content": raw,
                "mode": stat.S_IMODE(path.stat().st_mode), "diff": diff})

    @property
    def diff(self):
        return "\n".join(file["diff"] for file in self.files)

    def __enter__(self):
        if self.snapshot is not None:
            raise RuntimeError("a patch transaction cannot be reused")
        # Revalidate the complete set immediately before mutation, including
        # changes that occurred while a model was preparing its proposal.
        for file in self.files:
            target = workspace_path(self.root, file["path"], "patch target", must_exist=True, forbid_git=True)
            if hashlib.sha256(target.read_bytes()).hexdigest() != file["base_sha256"]:
                raise ValueError(f"base_sha256 changed before application: {file['path']}")
        self.snapshot = snapshot_workspace(self.root)
        try:
            # Stage every replacement first. os.replace avoids writes through
            # hardlinks, and any application failure restores the full snapshot.
            with tempfile.TemporaryDirectory(prefix=".autorepro-patch-", dir=self.root.parent) as staging:
                for index, file in enumerate(self.files):
                    staged = Path(staging) / str(index)
                    staged.write_bytes(file["content"])
                    staged.chmod(file["mode"])
                for index, file in enumerate(self.files):
                    target = workspace_path(self.root, file["path"], "patch target", must_exist=True, forbid_git=True)
                    if hashlib.sha256(target.read_bytes()).hexdigest() != file["base_sha256"]:
                        raise ValueError(f"base_sha256 changed during application: {file['path']}")
                    os.replace(Path(staging) / str(index), target)
                    self._invalidate_bytecode(file["path"])
        except BaseException as exc:
            self._rollback(exc)
            raise
        return self

    def commit(self):
        if self.snapshot is None:
            raise RuntimeError("the patch transaction has not started")
        self.committed = True

    def _rollback(self, original_error=None):
        try:
            self.restored, self.removed = restore_snapshot(self.root, self.snapshot)
            for file in self.files:
                self._invalidate_bytecode(file["path"])
        except BaseException as exc:
            self.rollback_error = f"{type(exc).__name__}: {exc}"
            if original_error is not None:
                original_error.add_note("Candidate rollback failed: " + self.rollback_error)
                return  # __enter__/__exit__ propagate the original exception.
            raise

    def _invalidate_bytecode(self, relative):
        source = Path(relative)
        if source.suffix.lower() != ".py":
            return
        directory = workspace_path(self.root, source.parent / "__pycache__", "bytecode cache", forbid_git=True)
        candidates = list(directory.iterdir()) if directory.is_dir() else []
        candidates = [file for file in candidates if file.name.startswith(source.stem + ".")
                      and file.suffix == ".pyc"]
        candidates.extend(self.root / source.with_suffix(extension) for extension in (".pyc", ".pyo"))
        for candidate in candidates:
            checked = workspace_path(self.root, candidate.relative_to(self.root), "bytecode cache", forbid_git=True)
            if checked.is_file():
                checked.unlink()

    def __exit__(self, error_type, error, traceback):
        if error is not None or not self.committed:
            self._rollback(error)
        return False

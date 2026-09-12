"""Task ingestion — turns CLI input into the Task object the rest of the
pipeline consumes. See spec Section 2 (Task Ingestion)."""
from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Task:
    id: str
    description: str
    target_paths: list
    diff: Optional[str]
    files_touched: int
    lines_changed: int
    cross_file_dependencies: bool
    is_retry: bool
    retry_count: int
    verifier_command: Optional[str]
    created_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "target_paths": self.target_paths,
            "diff": self.diff,
            "files_touched": self.files_touched,
            "lines_changed": self.lines_changed,
            "cross_file_dependencies": self.cross_file_dependencies,
            "is_retry": self.is_retry,
            "retry_count": self.retry_count,
            "verifier_command": self.verifier_command,
            "created_at": self.created_at,
        }


class IngestionError(Exception):
    """Raised when a task can't be assembled from the CLI input given."""


def _run_git(args: list, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise IngestionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _collect_path_mode(path: Path):
    """Path mode: council reads current file(s) as context; no diff exists
    yet — one gets generated later during the drafting phase."""
    if not path.exists():
        raise IngestionError(f"target path does not exist: {path}")

    if path.is_file():
        target_paths = [str(path)]
    else:
        target_paths = [str(p) for p in path.rglob("*") if p.is_file()]

    if not target_paths:
        raise IngestionError(f"no files found under: {path}")

    files_touched = len(target_paths)
    lines_changed = 0  # nothing changed yet in path mode
    return target_paths, files_touched, lines_changed


def _collect_diff_mode(ref: str, cwd: Path):
    """Diff mode: ingest an existing git diff instead of live files."""
    diff_text = _run_git(["diff", ref], cwd)
    if not diff_text.strip():
        raise IngestionError(f"git diff against '{ref}' produced no changes")

    stat_text = _run_git(["diff", "--stat", ref], cwd)
    target_paths = []
    lines_changed = 0
    for line in stat_text.splitlines():
        if "|" in line:
            file_part, change_part = line.split("|", 1)
            target_paths.append(file_part.strip())
            digits = "".join(c for c in change_part if c.isdigit())
            if digits:
                lines_changed += int(digits)

    return diff_text, target_paths, len(target_paths), lines_changed


def _detect_cross_file_dependencies(target_paths: list) -> bool:
    """Cheap heuristic for the classifier (Section 6): touching more than one
    file at once counts as cross-file until real import-graph analysis
    replaces this. Deliberately dumb — see the classifier's design notes."""
    return len(target_paths) > 1


def build_task(
    *,
    path: Optional[str],
    diff_ref: Optional[str],
    description: str,
    verifier_command: Optional[str],
    cwd: Optional[Path] = None,
    retry_count: int = 0,
) -> Task:
    """Assemble a Task from CLI input.

    Exactly one of path/diff_ref must be given — ingestion requires an
    explicit target and an explicit description. It never infers intent
    from a bare diff (see Task Ingestion, Section 2): the classifier's
    keyword signals need real text to scan, and guessing intent would mean
    burning a model call just to bootstrap the thing meant to save calls.
    """
    cwd = cwd or Path.cwd()

    if not description or not description.strip():
        raise IngestionError(
            "a --task description is required — ingestion doesn't infer "
            "intent from a diff or file alone (see Task Ingestion, Section 2)"
        )

    if bool(path) == bool(diff_ref):
        raise IngestionError("provide exactly one of: a target path, or --diff <ref>")

    if diff_ref:
        diff_text, target_paths, files_touched, lines_changed = _collect_diff_mode(diff_ref, cwd)
    else:
        diff_text = None
        p = Path(path)
        if not p.is_absolute():
            p = cwd / p
        target_paths, files_touched, lines_changed = _collect_path_mode(p)

    return Task(
        id=str(uuid.uuid4()),
        description=description.strip(),
        target_paths=target_paths,
        diff=diff_text,
        files_touched=files_touched,
        lines_changed=lines_changed,
        cross_file_dependencies=_detect_cross_file_dependencies(target_paths),
        is_retry=retry_count > 0,
        retry_count=retry_count,
        verifier_command=verifier_command,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

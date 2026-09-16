"""Bake-off harness — same prompt to N models, verify each in isolation.

Safety: the working tree is never touched. Each model's diff is applied
to a temp copy of the project (`git apply`), and the verifier runs there.
A model that chats instead of diffing is recorded as NO_DIFF — not a
verifier failure, just a non-applicable answer.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from council import executor as _executor
from council import ratelimit as _ratelimit
from council.config import CouncilConfig
from council.registry import ModelEntry
from council.task import Task
from council.verifier import run_verifier

MAX_CONTEXT_FILES = 10
MAX_TOTAL_CONTEXT_CHARS = 60000


@dataclass
class CompareRow:
    model_id: str
    cost_tier: str
    call_ok: bool
    diff_found: bool
    verifier_pass: Optional[bool]  # None when not applicable (no diff / call failed)
    verify_error: bool = False
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    error: Optional[str] = None

    @property
    def verdict(self) -> str:
        if not self.call_ok:
            return "CALL_FAIL"
        if not self.diff_found:
            return "NO_DIFF"
        if self.verify_error:
            return "VERIFY_ERROR"
        return "PASS" if self.verifier_pass else "FAIL"


def collect_context(task: Task, project_dir: Path) -> Dict[str, str]:
    """Read target files for prompt context. Missing/unreadable files are
    skipped — a thinner prompt beats a crashed harness."""
    context: Dict[str, str] = {}
    total = 0
    for raw in task.target_paths[:MAX_CONTEXT_FILES]:
        p = Path(raw)
        if not p.is_absolute():
            p = project_dir / p
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        try:
            rel = str(p.relative_to(project_dir))
        except ValueError:
            rel = p.name
        budget = MAX_TOTAL_CONTEXT_CHARS - total
        if budget <= 0:
            break
        context[rel] = text[:budget]
        total += len(context[rel])
    return context


def _slug(model_id: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_id)


def _normalize_diff_paths(diff: str) -> str:
    """Rewrite `---`/`+++` header paths with forward slashes. Models on a
    Windows-flavored streak emit `a/council\\task.py`; git apply won't
    resolve that to `council/task.py`. Body lines are left untouched."""
    out = []
    for line in diff.splitlines():
        if line.startswith(("--- ", "+++ ")) and "\\" in line:
            out.append(line.replace("\\", "/"))
        else:
            out.append(line)
    return "\n".join(out)


def _apply_diff(workdir: Path, diff: str) -> Optional[str]:
    """Apply a unified diff inside workdir. Returns None on success, else
    the error output (model produced a non-applicable diff).

    Tolerant by design: models miscount hunk headers constantly, so a
    strict apply is retried with `git apply --recount` (recompute counts
    from the hunk body). A diff whose *content* doesn't match the file is
    still a FAIL — and correctly so.
    """
    if shutil.which("git") is None:
        return "git not found on PATH — cannot apply diff"
    diff = _normalize_diff_paths(diff)
    if not diff.endswith("\n"):
        # Model outputs often drop the trailing newline; git apply
        # rejects the hunk as corrupt without it.
        diff += "\n"
    attempts = [
        ["git", "apply", "--whitespace=fix", "-"],
        ["git", "apply", "--whitespace=fix", "--recount", "-"],
    ]
    err = "git apply failed"
    for cmd in attempts:
        proc = subprocess.run(
            cmd,
            input=diff,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return None
        err = (proc.stderr or proc.stdout or "git apply failed").strip()[:500]
    return err


@dataclass
class CompareReport:
    compare_id: str
    task_description: str
    rows: List[CompareRow] = field(default_factory=list)
    raw_responses: Dict[str, str] = field(default_factory=dict)
    codes: Dict[str, str] = field(default_factory=dict)  # latest verified code per model


def _fresh_copy(project_dir: Path) -> Path:
    """Copy the project (minus VCS, caches, prior results) into a temp dir
    the harness may freely write into. The working tree is never touched."""
    tmpdir = Path(tempfile.mkdtemp(prefix="council-compare-"))
    for item in project_dir.iterdir():
        if item.name in (".git", ".council", "__pycache__", ".pytest_cache"):
            continue
        dest = tmpdir / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", ".pytest_cache"))
            else:
                shutil.copy2(item, dest)
        except OSError:
            continue
    return tmpdir


def run_create(models: List[ModelEntry], filename: str, task_description: str,
               project_dir: Path, config: CouncilConfig, prompt: str,
               timeout_seconds: int = 180,
               max_workers: int = 4) -> CompareReport:
    """Greenfield bake-off: each model writes a whole file. The file is
    saved into an isolated temp copy of the project and the verifier runs
    there (e.g. `python -m py_compile <file>`). A response with no code
    block is NO_DIFF, not a verifier failure."""
    compare_id = (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                  + "-" + uuid.uuid4().hex[:6])
    report = CompareReport(compare_id=compare_id, task_description=task_description)
    workers = max(1, min(len(models), max_workers))

    def _call(m: ModelEntry):
        _ratelimit.get_limiter().acquire(m)
        return m, _executor.safe_call(m, prompt, timeout_seconds,
                                          workdir=project_dir)

    calls: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for model, res in pool.map(_call, models):
            calls[model.id] = (model, res)
            report.raw_responses[model.id] = res.text if res.ok else ""

    for model in models:
        _, res = calls[model.id]
        if not res.ok:
            report.rows.append(CompareRow(
                model.id, model.cost_tier, False, False, None, error=res.error))
            continue
        code = _executor.extract_code_block(res.text)
        if not code:
            report.rows.append(CompareRow(
                model.id, model.cost_tier, True, False, None,
                latency_seconds=res.latency_seconds,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                est_cost_usd=res.estimated_cost_usd(
                    model.cost_per_1k_input, model.cost_per_1k_output)))
            continue
        tmpdir = _fresh_copy(project_dir)
        try:
            target = tmpdir / Path(filename).name  # basename only: never escape tmp
            target.write_text(code if code.endswith("\n") else code + "\n",
                              encoding="utf-8")
            workdir = ((tmpdir / config.working_dir).resolve()
                       if config.working_dir != "." else tmpdir)
            vreport = run_verifier(config, workdir)
            failing = vreport.failing_step()
            err = None
            if failing is not None:
                err = (failing.stderr or failing.stdout or failing.error or "")[:300]
            report.rows.append(CompareRow(
                model.id, model.cost_tier, True, True, vreport.overall_pass,
                verify_error=vreport.verify_error,
                latency_seconds=res.latency_seconds,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                est_cost_usd=res.estimated_cost_usd(
                    model.cost_per_1k_input, model.cost_per_1k_output),
                error=err or None))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return report


def run_compare(models: List[ModelEntry], task: Task, project_dir: Path,
               config: CouncilConfig, prompt: str,
               timeout_seconds: int = 120,
               max_workers: int = 4) -> CompareReport:
    compare_id = (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                  + "-" + uuid.uuid4().hex[:6])
    report = CompareReport(compare_id=compare_id, task_description=task.description)
    workers = max(1, min(len(models), max_workers))

    # Parallel fan-out for the network-bound calls, then apply + verify
    # each response in an isolated temp copy (CPU-bound, fast).
    def _call(m: ModelEntry):
        _ratelimit.get_limiter().acquire(m)
        return m, _executor.safe_call(m, prompt, timeout_seconds,
                                          workdir=project_dir)

    calls: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for model, res in pool.map(_call, models):
            calls[model.id] = (model, res)
            report.raw_responses[model.id] = res.text if res.ok else ""

    for model in models:
        _, res = calls[model.id]
        if not res.ok:
            report.rows.append(CompareRow(
                model.id, model.cost_tier, False, False, None, error=res.error))
            continue
        # Reuse the already-fetched response: verify inline (CPU-bound, fast).
        diff = _executor.extract_diff(res.text)
        if not diff:
            report.rows.append(CompareRow(
                model.id, model.cost_tier, True, False, None,
                latency_seconds=res.latency_seconds,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                est_cost_usd=res.estimated_cost_usd(
                    model.cost_per_1k_input, model.cost_per_1k_output)))
            continue
        # Apply + verify in an isolated temp copy.
        tmpdir = Path(tempfile.mkdtemp(prefix="council-compare-"))
        try:
            for item in project_dir.iterdir():
                if item.name in (".git", ".council", "__pycache__", ".pytest_cache"):
                    continue
                dest = tmpdir / item.name
                try:
                    if item.is_dir():
                        shutil.copytree(item, dest, ignore=shutil.ignore_patterns(
                            ".git", "__pycache__", ".pytest_cache"))
                    else:
                        shutil.copy2(item, dest)
                except OSError:
                    continue
            apply_err = _apply_diff(tmpdir, diff)
            if apply_err is not None:
                report.rows.append(CompareRow(
                    model.id, model.cost_tier, True, True, False,
                    latency_seconds=res.latency_seconds,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                    est_cost_usd=res.estimated_cost_usd(
                        model.cost_per_1k_input, model.cost_per_1k_output),
                    error=f"diff did not apply: {apply_err}"))
                continue
            workdir = (tmpdir / config.working_dir).resolve() if config.working_dir != "." else tmpdir
            vreport = run_verifier(config, workdir)
            failing = vreport.failing_step()
            err = None
            if failing is not None:
                err = (failing.stderr or failing.stdout or failing.error or "")[:300]
            report.rows.append(CompareRow(
                model.id, model.cost_tier, True, True, vreport.overall_pass,
                verify_error=vreport.verify_error,
                latency_seconds=res.latency_seconds,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                est_cost_usd=res.estimated_cost_usd(
                    model.cost_per_1k_input, model.cost_per_1k_output),
                error=err or None))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return report


def write_compare_report(project_dir: Path, report: CompareReport,
                         subdir: str = "compares") -> Path:
    """Persist responses + scoreboard under .council/<subdir>/<id>/."""
    outdir = project_dir / ".council" / subdir / report.compare_id
    outdir.mkdir(parents=True, exist_ok=True)
    for model_id, text in report.raw_responses.items():
        (outdir / f"{_slug(model_id)}.md").write_text(
            text or "(no response)", encoding="utf-8")
    summary = {
        "compare_id": report.compare_id,
        "task": report.task_description,
        "rows": [
            {"model": r.model_id, "tier": r.cost_tier, "verdict": r.verdict,
             "latency_s": round(r.latency_seconds, 2),
             "prompt_tokens": r.prompt_tokens,
             "completion_tokens": r.completion_tokens,
             "est_cost_usd": round(r.est_cost_usd, 6),
             "error": r.error}
            for r in report.rows
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Compare {report.compare_id}", "",
             f"task: {report.task_description}", "",
             "| model | tier | verdict | latency | tokens (in/out) | est. cost |",
             "|---|---|---|---|---|---|"]
    for r in report.rows:
        lines.append(
            f"| {r.model_id} | {r.cost_tier} | {r.verdict} | "
            f"{r.latency_seconds:.1f}s | {r.prompt_tokens}/{r.completion_tokens} | "
            f"${r.est_cost_usd:.6f} |")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outdir

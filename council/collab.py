"""Council mode — minds talking to each other, not just racing.

Phase 1 (write):   every model writes, verified.
Phase 2 (review):  every model reviews every peer's code in prose.
Phase 3 (revise):  every model revises its OWN code given peer reviews +
                   verifier notes + human notes, verified again.
Winner: first revising PASS, else best Phase-1 PASS.

`run_revise_round` re-runs Phase 3 standalone so a human can direct
extra rounds (`collab --interactive`): type notes, the council revises,
repeat until empty input (max 5 rounds).

Every round is verified in an isolated temp copy; the working tree is
never touched. Transcripts land under .council/councils/<id>/.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from council import executor as _executor
from council.compare import (
    CompareReport,
    CompareRow,
    _fresh_copy,
    write_compare_report,
)
from council.config import CouncilConfig
from council.registry import ModelEntry
from council.verifier import run_verifier

MAX_INTERACTIVE_ROUNDS = 5


def _verify_code(code: str, filename: str, project_dir: Path,
                 config: CouncilConfig) -> tuple[bool, bool, Optional[str]]:
    """Save code into a temp project copy, run the verifier there.
    Returns (overall_pass, verify_error, notes)."""
    tmpdir = _fresh_copy(project_dir)
    try:
        target = tmpdir / Path(filename).name
        target.write_text(code if code.endswith("\n") else code + "\n",
                          encoding="utf-8")
        workdir = ((tmpdir / config.working_dir).resolve()
                   if config.working_dir != "." else tmpdir)
        report = run_verifier(config, workdir)
        failing = report.failing_step()
        notes = None
        if failing is not None:
            notes = (failing.stderr or failing.stdout
                     or failing.error or "")[:1000]
        return report.overall_pass, report.verify_error, notes
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _row_for(model: ModelEntry, res, filename: str, project_dir: Path,
             config: CouncilConfig) -> Tuple[CompareRow, Optional[str]]:
    """Verify one response, build its scoreboard row. Returns (row, code)."""
    if not res.ok:
        return CompareRow(
            model.id, model.cost_tier, False, False, None,
            error=res.error), None
    code = _executor.extract_code_block(res.text)
    if not code:
        return CompareRow(
            model.id, model.cost_tier, True, False, None,
            latency_seconds=res.latency_seconds,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            est_cost_usd=res.estimated_cost_usd(
                model.cost_per_1k_input, model.cost_per_1k_output)), None
    passed, verr, notes = _verify_code(code, filename, project_dir, config)
    return CompareRow(
        model.id, model.cost_tier, True, True, passed,
        verify_error=verr,
        latency_seconds=res.latency_seconds,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        est_cost_usd=res.estimated_cost_usd(
            model.cost_per_1k_input, model.cost_per_1k_output),
        error=notes or None), code


def _fan_out(models: List[ModelEntry], prompts: Dict[str, str],
             timeout_seconds: int, max_workers: int,
             workdir: Optional[Path] = None):
    """One parallel call per model. Returns [(model, result)]."""
    workers = max(1, min(len(models), max_workers))

    def _call(m: ModelEntry):
        return m, _executor.safe_call(m, prompts[m.id], timeout_seconds,
                                      workdir=workdir)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_call, models))


def _reviewer_order(models: List[ModelEntry], author_id: str) -> List[ModelEntry]:
    """Peers who may review this author's code, best-ranked first so a
    reviewer cap keeps the strongest critics, not arbitrary ones."""
    peers = [m for m in models if m.id != author_id]
    peers.sort(key=lambda m: (
        m.quality_score.value if m.quality_score.value is not None else -1,
        -m.cost_per_1k_input,
    ), reverse=True)
    return peers


def run_council(models: List[ModelEntry], filename: str, task_description: str,
               project_dir: Path, config: CouncilConfig,
               create_prompt: str, timeout_seconds: int = 180,
               max_workers: int = 4, human_notes: str = "",
               max_reviewers: int = 0) -> CompareReport:
    compare_id = (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                  + "-council-" + uuid4().hex[:6])
    report = CompareReport(compare_id=compare_id, task_description=task_description)

    # ---- Phase 1: everyone writes ----
    codes: Dict[str, str] = {}
    for model, res in _fan_out(
            models, {m.id: create_prompt for m in models},
            timeout_seconds, max_workers, workdir=project_dir):
        report.raw_responses[f"phase1-{model.id}"] = res.text if res.ok else ""
        if res.ok:
            report.raw_responses[model.id] = res.text
        row, code = _row_for(model, res, filename, project_dir, config)
        report.rows.append(row)
        if code:
            codes[model.id] = code
    if not codes:
        return report

    # ---- Phase 2: peer review exchange (best critics first, capped) ----
    reviews: Dict[str, List[str]] = {m.id: [] for m in models}
    if len(models) > 1:
        jobs: Dict[str, str] = {}  # "critic\x00author" -> review prompt
        for author in models:
            if author.id not in codes:
                continue
            critics = _reviewer_order(models, author.id)
            if max_reviewers > 0:
                critics = critics[:max_reviewers]
            for critic in critics:
                jobs[f"{critic.id}\x00{author.id}"] = (
                    _executor.build_review_prompt(
                        task_description, filename,
                        codes[author.id], author.id))
        by_id = {m.id: m for m in models}
        by_key = {key: by_id[key.split("\x00")[0]] for key in jobs}
        workers = max(1, min(len(jobs), max_workers))

        def _review_call(key: str):
            critic = by_key[key]
            return key, _executor.safe_call(critic, jobs[key], timeout_seconds)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, res in pool.map(_review_call, list(by_key)):
                critic_id, author_id = key.split("\x00")
                tag = f"review-{critic_id}-on-{author_id}"
                report.raw_responses[tag] = res.text if res.ok else ""
                if res.ok and res.text.strip():
                    reviews[author_id].append(
                        f"--- review by {critic_id} ---\n{res.text.strip()[:2000]}")

    # ---- Phase 3: everyone revises their OWN code ----
    phase1_notes = {r.model_id: (r.error or "") for r in report.rows
                    if r.model_id in codes}
    rows, codes = run_revise_round(
        models, codes, reviews, filename, task_description,
        project_dir, config, verifier_notes=phase1_notes,
        human_notes=human_notes,
        timeout_seconds=timeout_seconds, max_workers=max_workers,
        round_tag="phase2", report=report)
    if any(r.verdict == "PASS" for r in rows):
        for r in rows:
            r.model_id = f"{r.model_id} (rev)"
        report.rows = rows
    report.codes = codes
    return report


def run_revise_round(models: List[ModelEntry], codes: Dict[str, str],
                     peer_reviews: Dict[str, List[str]],
                     filename: str, task_description: str,
                     project_dir: Path, config: CouncilConfig,
                     verifier_notes: Optional[Dict[str, str]] = None,
                     human_notes: str = "",
                     timeout_seconds: int = 180,
                     max_workers: int = 4, round_tag: str = "revise",
                     report: Optional[CompareReport] = None
                     ) -> Tuple[List[CompareRow], Dict[str, str]]:
    """One revise round over each model's own latest code. Returns
    (rows, new_codes). Appends transcripts to report when given."""
    verifier_notes = verifier_notes or {}
    prompts = {}
    for m in models:
        if m.id not in codes:
            continue
        prompts[m.id] = _executor.build_critique_prompt(
            task_description, filename, codes[m.id],
            verifier_notes.get(m.id, ""),
            human_notes,
            peer_reviews="\n\n".join(peer_reviews.get(m.id, [])),
        )
    revisers = [m for m in models if m.id in prompts]
    rows: List[CompareRow] = []
    new_codes = dict(codes)
    for model, res in _fan_out(revisers, prompts, timeout_seconds,
                               max_workers, workdir=project_dir):
        if report is not None:
            tag = f"{round_tag}-{model.id}"
            report.raw_responses[tag] = res.text if res.ok else ""
            if res.ok:
                report.raw_responses[model.id] = res.text
        row, code = _row_for(model, res, filename, project_dir, config)
        rows.append(row)
        if code:
            new_codes[model.id] = code
    return rows, new_codes


def write_council_report(project_dir: Path, report: CompareReport) -> Path:
    """Same on-disk shape as a compare report, under councils/."""
    return write_compare_report(project_dir, report, subdir="councils")

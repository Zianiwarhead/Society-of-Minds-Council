"""Council mode — minds building on each other, not just racing.

Phase 1: every model writes (same as a create bake-off), verified.
Phase 2: every model receives the best Phase-1 code + verifier notes and
returns an improved whole file, verified again.
Winner: first Phase-2 PASS, else best Phase-1 PASS, else FAIL with notes.

Every round is verified in an isolated temp copy; the working tree is
never touched. Transcripts land under .council/councils/<id>/.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
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


def run_council(models: List[ModelEntry], filename: str, task_description: str,
               project_dir: Path, config: CouncilConfig,
               create_prompt: str, timeout_seconds: int = 180,
               max_workers: int = 4) -> CompareReport:
    compare_id = (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                  + "-council-" + uuid4().hex[:6])
    report = CompareReport(compare_id=compare_id, task_description=task_description)
    workers = max(1, min(len(models), max_workers))

    def _call(m: ModelEntry, prompt: str):
        return m, _executor.safe_call(m, prompt, timeout_seconds)

    # ---- Phase 1: everyone writes ----
    phase1: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [(m, create_prompt) for m in models]
        for model, res in pool.map(lambda j: _call(*j), jobs):
            phase1[model.id] = res
            report.raw_responses[f"phase1-{model.id}"] = res.text if res.ok else ""
            if res.ok:
                report.raw_responses[model.id] = res.text  # latest wins below

    phase1_code: Dict[str, str] = {}
    for model in models:
        res = phase1[model.id]
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
        passed, verr, notes = _verify_code(code, filename, project_dir, config)
        phase1_code[model.id] = code
        report.rows.append(CompareRow(
            model.id, model.cost_tier, True, True, passed,
            verify_error=verr,
            latency_seconds=res.latency_seconds,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            est_cost_usd=res.estimated_cost_usd(
                model.cost_per_1k_input, model.cost_per_1k_output),
            error=notes or None))

    # Best Phase-1 code becomes the shared starting point: first PASS,
    # else first code at all. Nobody to build on -> stop here.
    base_id, base_code, base_notes = None, None, ""
    for row in report.rows:
        if row.verdict == "PASS" and row.model_id in phase1_code:
            base_id, base_code = row.model_id, phase1_code[row.model_id]
            base_notes = row.error or ""
            break
    if base_code is None and phase1_code:
        base_id = next(iter(phase1_code))
        base_code = phase1_code[base_id]
    if base_code is None:
        return report

    # ---- Phase 2: everyone improves the shared base ----
    def _revise(m: ModelEntry):
        prompt = _executor.build_critique_prompt(
            task_description, filename, base_code, base_notes)
        return m, _executor.safe_call(m, prompt, timeout_seconds)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        revisions = list(pool.map(_revise, models))

    final_rows: List[CompareRow] = []
    for model, res in revisions:
        tag = f"phase2-{model.id}"
        report.raw_responses[tag] = res.text if res.ok else ""
        report.raw_responses[model.id] = res.text if res.ok else report.raw_responses.get(model.id, "")
        if not res.ok:
            final_rows.append(CompareRow(
                model.id, model.cost_tier, False, False, None, error=res.error))
            continue
        code = _executor.extract_code_block(res.text)
        if not code:
            final_rows.append(CompareRow(
                model.id, model.cost_tier, True, False, None,
                latency_seconds=res.latency_seconds,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                est_cost_usd=res.estimated_cost_usd(
                    model.cost_per_1k_input, model.cost_per_1k_output)))
            continue
        passed, verr, notes = _verify_code(code, filename, project_dir, config)
        final_rows.append(CompareRow(
            model.id, model.cost_tier, True, True, passed,
            verify_error=verr,
            latency_seconds=res.latency_seconds,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            est_cost_usd=res.estimated_cost_usd(
                model.cost_per_1k_input, model.cost_per_1k_output),
            error=notes or None))

    # Winner replaces rows: Phase-2 PASS wins, else the Phase-1 story
    # stands (revisions stay in the transcripts for inspection).
    if any(r.verdict == "PASS" for r in final_rows):
        for r in final_rows:
            r.model_id = f"{r.model_id} (rev)"
        report.rows = final_rows
    return report


def write_council_report(project_dir: Path, report: CompareReport) -> Path:
    """Same on-disk shape as a compare report, under councils/."""
    return write_compare_report(project_dir, report, subdir="councils")

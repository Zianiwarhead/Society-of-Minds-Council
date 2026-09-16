"""CLI entry point.

    council run <path> --task "description"
    council run --diff <ref> --task "description"

See spec Sections 2 (Task Ingestion), 6 (Classifier), 3 (Verifier),
7 (Human Review Handoff). This wires config loading + task ingestion +
registry/model selection + verifier run + escalation decision together —
and now the actual model-calling loop: on a verify failure, `run` calls
the next model, applies its diff in an isolated copy (never the working
tree — see `_fresh_copy`/`_apply_diff` from compare.py), verifies there,
and loops via `decide_next` until DONE, VERIFY_ERROR, or HUMAN_REVIEW.

Nothing is auto-applied to the working tree by default (Section 7): a
passing diff is written to `.council/runs/<task-id>/final.diff` for you
to review and apply yourself. Pass --apply to have it applied directly.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from council import executor as _executor
from council.classifier import (
    Attempt,
    NoEligibleModelError,
    TaskState,
    classify_complexity,
    complexity_score,
    decide_next,
    required_capabilities,
    select_initial_model,
)
from council.compare import _apply_diff, _fresh_copy, collect_context
from council.config import ConfigError, VerifyStep, load_config
from council.discovery import (
    DiscoveryError,
    load_merged_registry,
    sync_openrouter,
)
from council.task import IngestionError, Task, build_task
from council.registry import ModelEntry, ModelRegistry, RegistryError, load_registry
from council.verifier import VerifyReport, run_verifier

MAX_LOOP_ATTEMPTS = 6  # decide_next's own state machine terminates well before
                       # this (free -> free-retry/escalate -> paid -> HUMAN_REVIEW
                       # is 3 attempts); this is a safety net, not the real limit.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="council",
        description=(
            "AI Council — multi-model task routing with free-first, "
            "escalate-on-failure execution."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Ingest a task and hand it to the council.")
    run.add_argument(
        "path",
        nargs="?",
        help="File or directory to target (path mode). Omit if using --diff.",
    )
    run.add_argument(
        "--diff",
        metavar="REF",
        help="Git ref to diff against instead of a live path (diff mode), e.g. HEAD~1.",
    )
    run.add_argument(
        "--task",
        required=True,
        metavar="DESCRIPTION",
        help='What the council should do, e.g. --task "add input validation to login".',
    )
    run.add_argument(
        "--verify",
        metavar="COMMAND",
        help="Override the project's configured verify steps for this run only.",
    )
    run.add_argument(
        "--project-dir",
        default=".",
        metavar="PATH",
        help="Project root containing .council.yaml. Defaults to the current directory.",
    )
    run.add_argument(
        "--registry",
        default="models.yaml",
        metavar="PATH",
        help="Path to models.yaml. Defaults to ./models.yaml (falls back to <project-dir>/models.yaml).",
    )
    run.add_argument(
        "--live",
        action="store_true",
        help="Merge the cached live catalog (see `models sync`) under models.yaml overrides.",
    )
    run.add_argument(
        "--cache-dir",
        default=None,
        metavar="PATH",
        help="Cache dir for the live catalog. Defaults to ~/.cache/council.",
    )
    run.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Per-model call timeout. Defaults to 120.",
    )
    run.add_argument(
        "--apply",
        action="store_true",
        help="Apply the final passing diff to the working tree. Default: "
             "write it to .council/runs/<task-id>/final.diff and leave your "
             "tree untouched — nothing is auto-applied unless you opt in.",
    )

    models = subparsers.add_parser("models", help="Inspect the model registry.")
    models.add_argument(
        "--registry",
        default="models.yaml",
        metavar="PATH",
        help="Path to models.yaml. Defaults to ./models.yaml.",
    )
    models.add_argument(
        "--live",
        action="store_true",
        help="Merge the cached live catalog under models.yaml overrides.",
    )
    models.add_argument(
        "--cache-dir",
        default=None,
        metavar="PATH",
        help="Cache dir for the live catalog. Defaults to ~/.cache/council.",
    )
    models_sub = models.add_subparsers(dest="models_command", required=False)
    sync = models_sub.add_parser("sync", help="Refresh the cached live model catalog.")
    sync.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter"],
        help="Provider catalog to sync. v1: openrouter only.",
    )
    sync.add_argument(
        "--refresh",
        action="store_true",
        help="Force a network fetch even if the cache is fresh.",
    )
    sync.add_argument(
        "--cache-dir",
        default=None,
        metavar="PATH",
        help="Cache dir for the live catalog. Defaults to ~/.cache/council.",
    )

    compare = subparsers.add_parser(
        "compare",
        help="Bake-off: same prompt to N models, verify each in isolation.",
    )
    compare.add_argument(
        "path",
        nargs="?",
        help="File or directory to target (also used for prompt context).",
    )
    compare.add_argument(
        "--task",
        required=True,
        metavar="DESCRIPTION",
        help='The task every model attempts, e.g. --task "fix the off-by-one".',
    )
    compare.add_argument(
        "--models",
        default=None,
        metavar="IDS",
        help="Comma-separated model ids. Defaults to all available free-tier models.",
    )
    compare.add_argument(
        "--tier",
        default="free",
        choices=["free", "paid", "hybrid", "all"],
        help="Model pool when --models is omitted. Defaults to free.",
    )
    compare.add_argument(
        "--project-dir",
        default=".",
        metavar="PATH",
        help="Project root containing .council.yaml. Defaults to the current directory.",
    )
    compare.add_argument(
        "--registry",
        default="models.yaml",
        metavar="PATH",
        help="Path to models.yaml. Defaults to ./models.yaml (falls back to <project-dir>/models.yaml).",
    )
    compare.add_argument(
        "--live",
        action="store_true",
        help="Merge the cached live catalog (see `models sync`) under models.yaml overrides.",
    )
    compare.add_argument(
        "--cache-dir",
        default=None,
        metavar="PATH",
        help="Cache dir for the live catalog. Defaults to ~/.cache/council.",
    )
    compare.add_argument(
        "--create",
        default=None,
        metavar="FILE",
        help="Greenfield mode: models write a whole new FILE (e.g. game.py) "
             "instead of a diff. The verifier runs against it in isolation.",
    )
    compare.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Per-model call timeout. Defaults to 120.",
    )

    collab = subparsers.add_parser(
        "collab",
        help="Council mode: models write, critique, and revise together.",
    )
    collab.add_argument(
        "--create",
        required=True,
        metavar="FILE",
        help="File the council builds together, e.g. game.py.",
    )
    collab.add_argument(
        "--task",
        required=True,
        metavar="DESCRIPTION",
        help="What the council should build.",
    )
    collab.add_argument(
        "--models",
        default=None,
        metavar="IDS",
        help="Comma-separated model ids. Defaults to all available free-tier models.",
    )
    collab.add_argument(
        "--tier",
        default="free",
        choices=["free", "paid", "hybrid", "all"],
        help="Model pool when --models is omitted. Defaults to free.",
    )
    collab.add_argument(
        "--project-dir",
        default=".",
        metavar="PATH",
        help="Project root containing .council.yaml. Defaults to the current directory.",
    )
    collab.add_argument(
        "--registry",
        default="models.yaml",
        metavar="PATH",
        help="Path to models.yaml. Defaults to ./models.yaml (falls back to <project-dir>/models.yaml).",
    )
    collab.add_argument(
        "--live",
        action="store_true",
        help="Merge the cached live catalog (see `models sync`) under models.yaml overrides.",
    )
    collab.add_argument(
        "--cache-dir",
        default=None,
        metavar="PATH",
        help="Cache dir for the live catalog. Defaults to ~/.cache/council.",
    )
    collab.add_argument(
        "--timeout",
        type=int,
        default=180,
        metavar="SECONDS",
        help="Per-model call timeout. Defaults to 180 (two rounds of long outputs).",
    )
    collab.add_argument(
        "--feedback",
        default=None,
        metavar="FILE",
        help="Your playtest notes (plain text). Fed to the revise round as "
             "human judgment the verifier can't provide.",
    )
    collab.add_argument(
        "--interactive",
        action="store_true",
        help="After the scoreboard, direct extra revise rounds yourself: "
             "type notes, the council revises, repeat. Empty line = done.",
    )
    collab.add_argument(
        "--max-reviewers",
        type=int,
        default=0,
        metavar="N",
        help="Cap reviewers per author (best-ranked first). 0 = every peer "
             "reviews. Needed past a handful of minds (pairs scale as N^2).",
    )

    return parser


def _load_registry_for_run(
    registry_raw: str, project_dir: Path, live: bool, cache_dir_raw: Optional[str]
) -> tuple[ModelRegistry, Optional[str]]:
    """Returns (registry, live_note). live_note describes the merge for logging."""
    cache_dir = Path(cache_dir_raw) if cache_dir_raw else None
    if not live:
        return load_registry(_resolve_registry_path(registry_raw, project_dir)), None
    merged, n_new, n_total, stale = load_merged_registry(
        _resolve_registry_path(registry_raw, project_dir), cache_dir
    )
    note = f"live merge: +{n_new} discovered, {n_total} total{' (cache STALE)' if stale else ''}"
    return merged, note


def _resolve_registry_path(raw: str, project_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    candidate = project_dir / raw
    if candidate.exists():
        return candidate
    return p  # let load_registry raise the "not found" error


def _write_review(
    project_dir: Path,
    task_id: str,
    description: str,
    model_id: str,
    report: VerifyReport,
    state: TaskState,
    reason: str,
    environment_error: bool,
    diffs: Optional[List[tuple]] = None,  # [(model_id, diff_text), ...] in attempt order
) -> Path:
    """Write the Section 7 review record. Never raises — a failed handoff
    must not mask the original verifier result. `diffs`, when given,
    preserves every attempted diff alongside the summary — nothing gets
    lost just because it didn't pass."""
    review_dir = project_dir / ".council" / "reviews" / task_id
    try:
        review_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Council review — {task_id}",
            "",
            f"task: {description}",
            f"model: {model_id}",
            f"state: {state.value}",
            f"environment_error: {str(environment_error).lower()}",
            f"reason: {reason}",
            "",
            "## Steps",
        ]
        for step in report.steps:
            lines.append(
                f"- {step.name}: {'PASS' if step.passed else 'FAIL'}"
                f" (required={step.required}"
                f"{', timed_out' if step.timed_out else ''}"
                f"{', error=' + step.error if step.error else ''})"
            )
        failing = report.failing_step()
        if failing is not None:
            lines += ["", f"failing_step: {failing.name}"]
        if diffs:
            lines += ["", "## Attempts"]
            for i, (attempt_model_id, _) in enumerate(diffs, start=1):
                lines.append(f"- attempt-{i}.diff — {attempt_model_id}")
        (review_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for i, step in enumerate(report.steps, start=1):
            log = (
                f"$ {step.name}\n"
                f"passed: {step.passed}\n"
                f"required: {step.required}\n"
                f"timed_out: {step.timed_out}\n"
                f"error: {step.error}\n"
                f"--- stdout ---\n{step.stdout}\n"
                f"--- stderr ---\n{step.stderr}\n"
            )
            (review_dir / f"verify-log-{i}.txt").write_text(log, encoding="utf-8")
        if diffs:
            for i, (_, diff_text) in enumerate(diffs, start=1):
                (review_dir / f"attempt-{i}.diff").write_text(diff_text, encoding="utf-8")
    except OSError as exc:
        print(f"council: warning: could not write review record: {exc}", file=sys.stderr)
    return review_dir


def _run_model_attempt(
    model: ModelEntry, task: Task, project_dir: Path,
    config, timeout_seconds: int,
) -> tuple:
    """Call one model for a fix, apply its diff in an isolated copy (the
    working tree is never touched here — same guarantee as compare/collab),
    verify there. Returns (Attempt, VerifyReport | None, diff | None, error | None).

    Model-attributable failures (call error, no diff, unappliable diff)
    are plain FAILs, not VERIFY_ERRORs: a VERIFY_ERROR would short-circuit
    `decide_next` straight to the terminal state instead of escalating to
    the next model, which is exactly what the loop exists to do.
    """
    context = collect_context(task, project_dir)
    prompt = _executor.build_prompt(task.description, context)
    res = _executor.safe_call(model, prompt, timeout_seconds, workdir=project_dir)
    if not res.ok:
        return (
            Attempt(model_id=model.id, passed=False, verify_error=False),
            None, None, res.error,
        )

    diff = _executor.extract_diff(res.text)
    if not diff:
        return (
            Attempt(model_id=model.id, passed=False, verify_error=False),
            None, None, "model did not return a diff",
        )

    tmpdir = _fresh_copy(project_dir)
    try:
        apply_err = _apply_diff(tmpdir, diff)
        if apply_err is not None:
            return (
                Attempt(model_id=model.id, passed=False, verify_error=False),
                None, diff, f"diff did not apply: {apply_err}",
            )
        workdir = (tmpdir / config.working_dir).resolve() if config.working_dir != "." else tmpdir
        report = run_verifier(config, workdir)
        attempt = Attempt(
            model_id=model.id, passed=report.overall_pass, verify_error=report.verify_error
        )
        return attempt, report, diff, None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _cmd_models(args: argparse.Namespace) -> int:
    if getattr(args, "models_command", None) == "sync":
        if args.provider != "openrouter":
            print(f"council: unsupported provider '{args.provider}'", file=sys.stderr)
            return 2
        try:
            result = sync_openrouter(
                cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                refresh=args.refresh,
            )
        except DiscoveryError as exc:
            print(f"council: sync error: {exc}", file=sys.stderr)
            return 2
        print(f"synced {result.total} model(s) from openrouter "
              f"({result.free} free, {result.paid} paid)")
        print(f"cache: {result.cache_path}"
              + (" (served from cache)" if result.cached else ""))
        if result.stale:
            print("warning: cache is stale", file=sys.stderr)
        return 0

    try:
        if getattr(args, "live", False):
            cache_dir = Path(args.cache_dir) if args.cache_dir else None
            registry, n_new, n_total, stale = load_merged_registry(
                Path(args.registry), cache_dir
            )
            print(f"{n_total} model(s) loaded from {args.registry} "
                  f"(+{n_new} discovered from live cache"
                  + ("; cache STALE" if stale else "") + ")")
        else:
            registry = load_registry(Path(args.registry))
            print(f"{len(registry.all())} model(s) loaded from {args.registry}")
    except (RegistryError, DiscoveryError) as exc:
        print(f"council: registry error: {exc}", file=sys.stderr)
        return 2

    all_models = registry.all()
    missing = registry.missing_keys()

    print()
    for m in all_models:
        key_status = "key set" if m.has_api_key() else "KEY MISSING"
        print(f"  {m.id}  [{m.cost_tier}]  ({key_status})")
        print(f"    provider        : {m.provider}")
        print(f"    capabilities    : {m.capabilities}")
        print(f"    not_trusted_for : {m.not_trusted_for}")
        print(f"    quality_score   : {m.quality_score.value} (source: {m.quality_score.source})")
        print(f"    escalates_to    : {m.escalates_to}")
        print()

    if missing:
        print(f"missing env vars for {len(missing)} key(s): {missing}", file=sys.stderr)

    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()

    try:
        config = load_config(project_dir)
    except ConfigError as exc:
        print(f"council: config error: {exc}", file=sys.stderr)
        return 2

    if args.verify:
        config.verify = [VerifyStep(name="override", run=args.verify, required=True)]

    verifier_command = config.verify[0].run if config.verify else None

    try:
        task = build_task(
            path=args.path,
            diff_ref=args.diff,
            description=args.task,
            verifier_command=verifier_command,
            cwd=project_dir,
        )
    except IngestionError as exc:
        print(f"council: {exc}", file=sys.stderr)
        return 2

    try:
        registry, live_note = _load_registry_for_run(
            args.registry, project_dir,
            getattr(args, "live", False),
            getattr(args, "cache_dir", None),
        )
    except (RegistryError, DiscoveryError) as exc:
        print(f"council: registry error: {exc}", file=sys.stderr)
        return 2

    caps = required_capabilities(task)
    complexity = classify_complexity(task)
    score = complexity_score(task)

    try:
        first_model = select_initial_model(registry, task)
    except NoEligibleModelError as exc:
        capable = registry.eligible_for(caps)
        missing = registry.missing_keys(capable) if capable else registry.missing_keys()
        print(f"council: no eligible model: {exc}", file=sys.stderr)
        if missing:
            print(f"council: missing env vars for capable models: {missing}", file=sys.stderr)
        return 2

    print(f"Task ingested: {task.id}")
    print(f"  description       : {task.description}")
    print(f"  target_paths      : {task.target_paths}")
    print(f"  files_touched     : {task.files_touched}")
    print(f"  lines_changed     : {task.lines_changed}")
    print(f"  cross_file_deps   : {task.cross_file_dependencies}")
    print(f"  complexity        : {complexity} (score {score})")
    print(f"  required_caps     : {caps}")
    print(f"  first_model       : {first_model.id} [{first_model.cost_tier}]")
    print(f"  sandbox.image     : {config.sandbox.image}")
    if live_note:
        print(f"  registry          : {live_note}")
    print()

    # Zero-cost check: does the current tree already pass, with no model
    # called at all? This is the free-first philosophy taken to its logical
    # end — don't spend a single token if nothing needs fixing.
    workdir = (project_dir / config.working_dir).resolve()
    as_is_report = run_verifier(config, workdir)
    for step in as_is_report.steps:
        status = "PASS" if step.passed else "FAIL"
        extra = f" error={step.error}" if step.error else ""
        print(f"  as-is verify {step.name}: {status}{extra}")

    if as_is_report.overall_pass:
        print("\ncouncil: DONE — verifier already passes on the current tree (no model called)")
        return 0

    print(f"\ncouncil: current tree fails verify — engaging {first_model.id}\n")

    attempts: List[Attempt] = []
    diffs: List[tuple] = []  # [(model_id, diff_text), ...] in attempt order
    current_model = first_model

    for _ in range(MAX_LOOP_ATTEMPTS):
        print(f"council: trying {current_model.id} [{current_model.cost_tier}]...")
        attempt, vreport, diff, call_err = _run_model_attempt(
            current_model, task, project_dir, config, args.timeout,
        )
        attempts.append(attempt)
        if diff:
            diffs.append((current_model.id, diff))

        if call_err:
            print(f"  -> {call_err}")
        elif vreport is not None:
            for step in vreport.steps:
                status = "PASS" if step.passed else "FAIL"
                extra = f" error={step.error}" if step.error else ""
                print(f"  verify {step.name}: {status}{extra}")

        try:
            next_step = decide_next(task, registry, attempts)
        except NoEligibleModelError as exc:
            fallback_report = vreport if vreport is not None else as_is_report
            review_dir = _write_review(
                project_dir, task.id, task.description, current_model.id,
                fallback_report, TaskState.HUMAN_REVIEW, str(exc), False, diffs,
            )
            print(f"council: escalation failed: {exc}", file=sys.stderr)
            print(f"council: HUMAN_REVIEW — see {review_dir}", file=sys.stderr)
            return 4

        if next_step.state == TaskState.DONE:
            final_model_id, final_diff = diffs[-1]
            out_dir = project_dir / ".council" / "runs" / task.id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "final.diff"
            out_path.write_text(final_diff, encoding="utf-8")
            print(f"\ncouncil: DONE — {final_model_id} passed verify.")
            print(f"council: diff written to {out_path}")
            if args.apply:
                apply_err = _apply_diff(project_dir, final_diff)
                if apply_err is not None:
                    print(f"council: --apply failed: {apply_err}", file=sys.stderr)
                    print("council: your working tree is unchanged; apply the diff manually.")
                    return 1
                print("council: applied to your working tree (--apply).")
            else:
                print("council: nothing was applied to your working tree — "
                      "review the diff and apply it yourself, or re-run with --apply.")
            return 0

        fallback_report = vreport if vreport is not None else as_is_report

        if next_step.state == TaskState.VERIFY_ERROR:
            review_dir = _write_review(
                project_dir, task.id, task.description, current_model.id,
                fallback_report, next_step.state, next_step.reason, True, diffs,
            )
            print(f"\ncouncil: VERIFY_ERROR — {next_step.reason}", file=sys.stderr)
            print(f"council: review record at {review_dir}", file=sys.stderr)
            return 3

        if next_step.state == TaskState.HUMAN_REVIEW:
            review_dir = _write_review(
                project_dir, task.id, task.description, current_model.id,
                fallback_report, next_step.state, next_step.reason, False, diffs,
            )
            print(f"\ncouncil: HUMAN_REVIEW — {next_step.reason}", file=sys.stderr)
            print(f"council: review record at {review_dir}", file=sys.stderr)
            return 4

        # RUNNING or ESCALATED with a next model — loop.
        print(f"  -> {next_step.state.value}: next is {next_step.model.id} ({next_step.reason})\n")
        current_model = next_step.model

    # Safety net only — decide_next's own state machine terminates in at
    # most 3 attempts, so this should never actually trigger.
    print("council: exceeded max attempts without resolution — treating as HUMAN_REVIEW", file=sys.stderr)
    review_dir = _write_review(
        project_dir, task.id, task.description, current_model.id,
        vreport if vreport is not None else as_is_report,
        TaskState.HUMAN_REVIEW, "exceeded max loop attempts", False, diffs,
    )
    print(f"council: review record at {review_dir}", file=sys.stderr)
    return 4


def _select_pool(registry: ModelRegistry, args: argparse.Namespace) -> tuple[list, Optional[str]]:
    """Resolve --models / --tier to available models. Returns (pool, error)."""
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        try:
            return [registry.get(m) for m in wanted], None
        except RegistryError as exc:
            return [], str(exc)
    if args.tier == "all":
        pool = registry.available()
    else:
        pool = registry.available(registry.eligible_for(["code_generation"], cost_tier=args.tier))
    if not pool:
        capable = (registry.eligible_for(["code_generation"])
                   if args.tier in ("all", None) else
                   registry.eligible_for(["code_generation"], cost_tier=args.tier))
        missing = registry.missing_keys(capable) if capable else registry.missing_keys()
        err = f"no available models in pool '{args.tier or args.models}'"
        if missing:
            err += f" (missing env vars: {missing})"
        return [], err
    return pool, None


def _print_scoreboard(report) -> None:
    print(f"{'model':<40} {'tier':<7} {'verdict':<12} {'latency':<8} {'tok in/out':<14} {'est cost'}")
    for r in report.rows:
        print(f"{r.model_id:<40} {r.cost_tier:<7} {r.verdict:<12} "
              f"{r.latency_seconds:<8.1f} {r.prompt_tokens}/{r.completion_tokens:<11} "
              f"${r.est_cost_usd:.6f}")
        if r.error:
            print(f"    -> {(r.error or '')[:160]}")


def _load_config_and_pool(args: argparse.Namespace):
    """Shared setup for compare/collab. Returns (project_dir, config,
    pool, live_note) or (None, exit_code) on error."""
    project_dir = Path(args.project_dir).resolve()
    try:
        config = load_config(project_dir)
    except ConfigError as exc:
        print(f"council: config error: {exc}", file=sys.stderr)
        return None, 2
    try:
        registry, live_note = _load_registry_for_run(
            args.registry, project_dir,
            getattr(args, "live", False),
            getattr(args, "cache_dir", None),
        )
    except (RegistryError, DiscoveryError) as exc:
        print(f"council: registry error: {exc}", file=sys.stderr)
        return None, 2
    pool, err = _select_pool(registry, args)
    if err:
        print(f"council: {err}", file=sys.stderr)
        return None, 2
    return (project_dir, config, pool, live_note), 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from council.compare import (
        collect_context,
        run_compare,
        run_create,
        write_compare_report,
    )
    from council.executor import build_create_prompt, build_prompt

    setup, code = _load_config_and_pool(args)
    if setup is None:
        return code
    project_dir, config, pool, live_note = setup

    if getattr(args, "create", None) and args.path:
        print("council: give either a target path or --create FILE, not both",
              file=sys.stderr)
        return 2

    if getattr(args, "create", None):
        # Greenfield mode: no target file, models write the whole thing.
        filename = Path(args.create).name
        prompt = build_create_prompt(args.task, filename)
        print(f"Bake-off (create {filename}): {len(pool)} model(s) — '{args.task}'")
        if live_note:
            print(f"  registry: {live_note}")
        print(f"  pool: {', '.join(m.id for m in pool)}")
        print()
        report = run_create(pool, filename, args.task, project_dir, config,
                            prompt, timeout_seconds=args.timeout)
    else:
        if not args.path:
            print("council: give a target path or --create FILE", file=sys.stderr)
            return 2
        try:
            task = build_task(
                path=args.path, diff_ref=None, description=args.task,
                verifier_command=config.verify[0].run if config.verify else None,
                cwd=project_dir,
            )
        except IngestionError as exc:
            print(f"council: {exc}", file=sys.stderr)
            return 2

        context = collect_context(task, project_dir)
        prompt = build_prompt(task.description, context)

        print(f"Bake-off: {len(pool)} model(s) — '{task.description}'")
        if live_note:
            print(f"  registry: {live_note}")
        print(f"  pool: {', '.join(m.id for m in pool)}")
        print()

        report = run_compare(pool, task, project_dir, config, prompt,
                             timeout_seconds=args.timeout)
    outdir = write_compare_report(project_dir, report)

    _print_scoreboard(report)
    print(f"\nreport: {outdir}")
    return 0 if any(r.verdict == "PASS" for r in report.rows) else 1


def _cmd_collab(args: argparse.Namespace) -> int:
    from council.collab import run_council, write_council_report
    from council.executor import build_create_prompt

    setup, code = _load_config_and_pool(args)
    if setup is None:
        return code
    project_dir, config, pool, live_note = setup

    if len(pool) < 1:
        print("council: collab needs at least one model", file=sys.stderr)
        return 2

    filename = Path(args.create).name
    prompt = build_create_prompt(args.task, filename)
    print(f"Council (create {filename}): {len(pool)} mind(s) — '{args.task}'")
    if live_note:
        print(f"  registry: {live_note}")
    print(f"  pool: {', '.join(m.id for m in pool)}")
    print("  rounds: write -> review -> revise -> best verified wins")
    human_notes = ""
    if getattr(args, "feedback", None):
        try:
            human_notes = Path(args.feedback).read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"council: cannot read feedback file: {exc}", file=sys.stderr)
            return 2
        if not human_notes:
            print("council: feedback file is empty", file=sys.stderr)
            return 2
        print(f"  human feedback: {args.feedback} ({len(human_notes)} chars)")
    print()

    report = run_council(pool, filename, args.task, project_dir, config,
                         prompt, timeout_seconds=args.timeout,
                         human_notes=human_notes,
                         max_reviewers=getattr(args, "max_reviewers", 0) or 0)
    outdir = write_council_report(project_dir, report)

    _print_scoreboard(report)
    print(f"\nreport: {outdir}")

    if getattr(args, "interactive", False):
        from council.collab import MAX_INTERACTIVE_ROUNDS, run_revise_round
        codes = dict(report.codes)
        for round_no in range(1, MAX_INTERACTIVE_ROUNDS + 1):
            try:
                notes = input(
                    f"\nDirect the council, round {round_no} "
                    f"(empty = done, games at {outdir}):\n> "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not notes:
                break
            rows, codes = run_revise_round(
                pool, codes, {}, filename, args.task, project_dir, config,
                human_notes=notes, timeout_seconds=args.timeout,
                round_tag=f"you-round{round_no}", report=report)
            if any(r.verdict == "PASS" for r in rows):
                for r in rows:
                    r.model_id = f"{r.model_id} (you{round_no})"
                report.rows = rows
            outdir = write_council_report(project_dir, report)
            _print_scoreboard(report)
            print(f"\nreport: {outdir}")

    return 0 if any(r.verdict == "PASS" for r in report.rows) else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "collab":
        return _cmd_collab(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI entry point.

    council run <path> --task "description"
    council run --diff <ref> --task "description"

See spec Sections 2 (Task Ingestion), 6 (Classifier), 3 (Verifier),
7 (Human Review Handoff). This wires config loading + task ingestion +
registry/model selection + verifier run + escalation decision together.

What it does NOT do yet: call an LLM to generate a diff. The verifier
runs against the current working tree as-is, so `run` today validates
the pipeline (ingest -> select -> verify -> decide next) without
pretending model codegen exists. Once provider clients land, the
RUNNING/ESCALATED next-step becomes a real retry with a new model
instead of a printed pointer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, List

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
from council.config import ConfigError, VerifyStep, load_config
from council.discovery import (
    DiscoveryError,
    load_merged_registry,
    sync_openrouter,
)
from council.task import IngestionError, build_task
from council.registry import ModelRegistry, RegistryError, load_registry
from council.verifier import VerifyReport, run_verifier


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
) -> Path:
    """Write the Section 7 review record. Never raises — a failed handoff
    must not mask the original verifier result."""
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
    except OSError as exc:
        print(f"council: warning: could not write review record: {exc}", file=sys.stderr)
    return review_dir


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
        model = select_initial_model(registry, task)
    except NoEligibleModelError as exc:
        # Fail fast with actionable output (Section 5): name the missing
        # env vars for the capable pool, not just "no model".
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
    print(f"  selected_model    : {model.id} [{model.cost_tier}]")
    print(f"  sandbox.image     : {config.sandbox.image}")
    if live_note:
        print(f"  registry          : {live_note}")
    print()

    workdir = (project_dir / config.working_dir).resolve()
    report = run_verifier(config, workdir)

    for step in report.steps:
        status = "PASS" if step.passed else "FAIL"
        extra = f" error={step.error}" if step.error else ""
        print(f"  verify {step.name}: {status}{extra}")

    attempt = Attempt(
        model_id=model.id,
        passed=report.overall_pass,
        verify_error=report.verify_error,
    )
    try:
        next_step = decide_next(task, registry, [attempt])
    except NoEligibleModelError as exc:
        print(f"council: escalation failed: {exc}", file=sys.stderr)
        review_dir = _write_review(
            project_dir, task.id, task.description, model.id,
            report, TaskState.HUMAN_REVIEW, str(exc), False,
        )
        print(f"council: HUMAN_REVIEW — see {review_dir}", file=sys.stderr)
        return 4

    if next_step.state == TaskState.DONE:
        print(f"\ncouncil: DONE — verifier passed on {model.id}")
        return 0

    if next_step.state == TaskState.VERIFY_ERROR:
        review_dir = _write_review(
            project_dir, task.id, task.description, model.id,
            report, next_step.state, next_step.reason, True,
        )
        print(f"\ncouncil: VERIFY_ERROR — {next_step.reason}", file=sys.stderr)
        print(f"council: review record at {review_dir}", file=sys.stderr)
        return 3

    if next_step.state == TaskState.HUMAN_REVIEW:
        review_dir = _write_review(
            project_dir, task.id, task.description, model.id,
            report, next_step.state, next_step.reason, False,
        )
        print(f"\ncouncil: HUMAN_REVIEW — {next_step.reason}", file=sys.stderr)
        print(f"council: review record at {review_dir}", file=sys.stderr)
        return 4

    # RUNNING or ESCALATED with a next model: no LLM codegen yet, so stop
    # here and point at the next step instead of looping.
    print(
        f"\ncouncil: verifier FAILED on {model.id} — next: "
        f"{next_step.state.value} with {next_step.model.id} ({next_step.reason})"
    )
    print(
        "(model codegen not implemented yet — apply a fix and re-run, "
        "or wire provider clients to retry automatically)"
    )
    return 1


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
    print("  rounds: write -> critique+revise -> best verified wins")
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
                         human_notes=human_notes)
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

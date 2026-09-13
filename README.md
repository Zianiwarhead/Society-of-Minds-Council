# Society of Minds Council — multi-model bake-offs, free-first routing

## Layout
- `council/task.py`      — task ingestion (path mode + diff mode → `Task` object)
- `council/config.py`    — loads `.council.yaml` (verify steps, sandbox settings)
- `council/registry.py`  — loads `models.yaml` overrides (small: your deltas only)
- `council/discovery.py` — live catalog cache + merge (`models sync`, `--live`)
- `council/executor.py`  — OpenRouter chat client (stdlib only, BYOK via env)
- `council/compare.py`   — bake-off harness (fan-out, isolated verify, scoreboard)
- `council/cli.py`       — `run` / `models` / `compare` entry points
- `tests/`               — pytest suite (`pytest tests/ -q`)

## Usage
```
pip install pyyaml

# path mode
python -m council.cli run src/login.py --task "add input validation" --project-dir .

# diff mode
python -m council.cli run --diff HEAD~1 --task "review this refactor" --project-dir .
```

Requires a `.council.yaml` in the project root:
```yaml
verify:
  - name: "lint"
    run: "ruff check ."
    required: true
sandbox:
  image: "python:3.11-slim"
```

## What this does and doesn't do
`council run` now wires the full local pipeline: ingest → classify →
select model → run verifier → escalation decision. Exit codes: `0` DONE,
`1` FAIL with a next model to try, `3` VERIFY_ERROR, `4` HUMAN_REVIEW
(both write `.council/reviews/<task-id>/` per spec Section 7).

What it still doesn't do: sandboxed (Docker) execution — the verifier
runs on host, and `compare` verifies in temp copies (never your tree).

## Bake-off (`council compare`)
Same prompt to N models, each diff applied + verified in isolation:

```
# one key covers most free + paid models
set OPENROUTER_API_KEY=...

python -m council.cli models sync --provider openrouter
python -m council.cli compare src/login.py --task "fix the off-by-one" --tier free
python -m council.cli compare src/login.py --task "..." --models "x/model-a,y/model-b"
```

Prints a scoreboard (verdict / latency / tokens / est. cost) and writes
`.council/compares/<id>/` with `summary.md`, `summary.json`, and each
model's raw response. Verdicts: `PASS` / `FAIL` / `NO_DIFF` (chatted
instead of diffing) / `CALL_FAIL` / `VERIFY_ERROR`.

## Council mode (`council collab`)
Minds racing (compare) vs. minds talking (collab):

```
python -m council.cli collab --create game.py --task "..." --models "a,b" --live
```

Round 1: everyone writes. Round 2: everyone reviews every peer's code
as numbered corrections (`review-<critic>-on-<author>.md`). Round 3:
everyone revises their *own* code, answering each correction
(`FIXED: <n>` / `KEPT: <n> because ...`). Best verified wins.
`--max-reviewers N` caps critics per author (pairs scale as N^2);
`--feedback notes.txt` / `--interactive` put you in the director's chair.

## OpenCode backend
Any model OpenCode knows can be a council mind — no extra keys, auth
stays in OpenCode:

```yaml
- id: "opencode-sonnet"
  provider: "opencode"
  endpoint: "anthropic/claude-sonnet-4-5"  # opencode model id
  api_key_env: "none"                      # opencode owns its auth
  backend: "opencode"
  cost_tier: "paid"
  capabilities: ["code_generation"]
```

Requires the `opencode` CLI on PATH. The council shells out to
`opencode run -m <provider/model> --format json` with the project dir
as cwd, so opencode minds can read your files like any agent session.
Mix backends freely: `--models "opencode-sonnet,nvidia/nemotron-3.5-lightning:free"`.

## Registry loader (`council/registry.py`)
Loads and validates `models.yaml` (see `models.yaml.example`):
- Required fields, valid `cost_tier` (`free`/`paid`/`hybrid`)
- `quality_score.source` restricted to `user_override`/`measured` — no scraper/judge sources (manual-only, per spec Section 8)
- `escalates_to` must reference a real model id, and escalation chains are checked for cycles
- Duplicate ids rejected

```
python -m council.cli models --registry models.yaml
```

`models.yaml` holds only your overrides — not the full catalog. The live
list comes from the provider handshake and merges underneath at runtime:

```
# refresh ~/.cache/council/openrouter-models.json (24h TTL, offline falls back to cache)
python -m council.cli models sync --provider openrouter

# list merged view / run against merged view
python -m council.cli models --registry models.yaml --live
python -m council.cli run src/login.py --task "..." --live
```

New IDs default to `capabilities=["code_generation"]`, unranked quality
(never wins ties), tier inferred from pricing (`0/0` → `free`).

Query primitives for the classifier to build on:
- `registry.eligible_for(required_caps, cost_tier=None)` — capability + exclusion filter
- `registry.available(models=None)` — filters to models whose API key is actually set (Section 5's fail-fast check)
- `registry.missing_keys(models=None)` — names the missing env vars for a candidate pool

## Classifier (`council/classifier.py`)
Complexity scoring, capability inference, model selection, and the
escalation state machine — all deterministic, no LLM calls:

- `classify_complexity(task)` — cheap scoring (files touched, lines changed,
  cross-file flag, retry flag, security keywords) → `simple`/`moderate`/`complex`
- `required_capabilities(task)` — derives what a model needs to be trusted
  for from the same signals
- `select_initial_model(registry, task)` — the free-first baseline: simple/
  moderate tasks start on the free tier, `complex` tasks skip straight to paid
- `decide_next(task, registry, attempts)` — the escalation state machine:
  `RUNNING → (fail) → RUNNING → (fail again) → ESCALATED → (fail) → HUMAN_REVIEW`,
  with a `VERIFY_ERROR` short-circuit that never counts toward the tally

Verified against 7 scenarios: simple-task free routing, complex/security task
skipping straight to paid, single-free-model retry-to-escalation, forced
paid escalation, paid failure → human review, verify-error short-circuit,
and the pass → done path.

## Verifier runner (`council/verifier.py`)
Executes the `.council.yaml` verify steps in order and produces the
PASS/FAIL/VERIFY_ERROR signal the classifier's escalation state machine
consumes:

- `run_verifier(config, workdir)` → `VerifyReport` with per-step
  `stdout`/`stderr` (needed for the escalation model to debug root cause,
  not just a bare boolean)
- A required step failing stops the run there; a non-required step failing
  is recorded but doesn't block `overall_pass`
- `VerifyReport.verify_error` is set — separately from `overall_pass` — when
  the step itself couldn't run at all: a timeout, or a missing/non-executable
  command. Missing commands are detected portably: Unix shells return
  126/127, but Windows `cmd.exe` returns 1, so the runner also checks the
  executable via `shutil.which` rather than trusting the exit code alone.

Verified against 5 cases: all steps passing, a required step failing (blocks
the rest), a non-required step failing (doesn't block), a missing command,
and a step that times out.

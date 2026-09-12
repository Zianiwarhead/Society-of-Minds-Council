# AI Council — Model Metadata Schema & Classifier Logic

## 1. Model Metadata Schema

Each model the council can call is described by one entry in a registry file (e.g. `models.yaml`). This is the source of truth the classifier reads to decide who gets a task.

```yaml
- id: "glm-4.6-coder"
  provider: "openrouter"          # openrouter | groq | google_ai_studio | anthropic | openai | local
  endpoint: "https://openrouter.ai/api/v1/chat/completions"
  api_key_env: "OPENROUTER_API_KEY"

  cost_tier: "free"               # free | paid | hybrid
  cost_per_1k_input: 0.0          # USD, 0.0 for free-tier
  cost_per_1k_output: 0.0

  capabilities:                   # what this model is trusted to do
    - "code_generation"
    - "boilerplate"
    - "refactor_simple"
  not_trusted_for:                # explicit exclusions override capability guesses
    - "security_audit"
    - "cross_file_architecture"

  context_window: 128000
  quality_score: 0.78             # 0-1, from your own benchmark harness, not vendor marketing
  latency_class: "fast"           # fast | medium | slow
  rate_limit_rpm: 20
  max_concurrent: 3

  escalates_to: "claude-sonnet-4-6"   # which model handles this one's failures
  last_verified: "2026-09-01"         # when someone last confirmed the endpoint/model actually works
```

Key design choices baked into the schema:

- **`cost_tier` is a field, not a hardcoded list** — this is what lets users (or the classifier) reclassify a model as their pricing/access changes, without touching code.
- **`capabilities` + `not_trusted_for` together** — a model can be great at boilerplate but explicitly excluded from security work, even if generically "capable." This stops the classifier from routing a fast free model into a job it's bad at just because nothing says it can't.
- **`quality_score` comes from your own harness**, not vendor claims — this is what should get updated as you run real tasks through the council and log pass/fail rates per model. Otherwise the "free models are comparable to paid" claim stays anecdotal instead of measured.
- **`escalates_to`** encodes the escalation graph directly in metadata, so the classifier doesn't need separate routing rules per model — it just follows the chain.
- **`last_verified`** matters because free-tier model availability on OpenRouter/Groq/etc. changes often; a stale entry silently breaking your pipeline is worse than no entry.

## 2. Task Ingestion

**v1 scope: CLI-only, explicit description, no auto-watching.**

```
council run <path> --task "add input validation to the login handler" [--verify "pytest tests/"]
council run --diff HEAD~1 --task "review this refactor for regressions"
```

Two entry modes, both requiring an explicit `--task` description:

- **Path mode** — points at a file or directory. The council reads current content as context and generates a diff against it.
- **Diff mode** — points at an existing git diff (`--diff <ref>`) instead of live files. Useful for reviewing a change that already exists rather than generating a new one.

**Deliberately deferred, not v1:**
- **Watch mode** (`council watch ./src` reacting to file saves) — real value later, but it multiplies false triggers and partial-edit noise before the core loop is even proven. Not worth the complexity yet.
- **Stdin/pipe ingestion** for CI scripting — easy to bolt on once the CLI path works; no reason to design it before the core loop exists.

**Why require an explicit `--task` description instead of inferring intent from the diff alone:** the classifier's cheap keyword signals (`security`, `auth`, `CVE`, etc. — see below) depend on having real text to scan. Inferring "what is this task" from a bare diff would need an LLM call just to bootstrap the thing meant to save LLM calls. Requiring the user to state intent up front keeps the classifier's first pass free and deterministic.

### Ingested Task object
This is what ingestion hands to the classifier:

```yaml
task:
  id: "uuid"
  description: "add input validation to the login handler"   # from --task
  target_paths: ["src/auth/login.py"]
  diff: null                          # populated only in --diff mode
  files_touched: 1                    # derived
  lines_changed: 0                    # derived; 0 until council generates a diff
  cross_file_dependencies: false      # derived heuristic, e.g. import graph fan-out
  is_retry: false
  retry_count: 0
  verifier_command: "pytest tests/"   # from --verify flag or project config (see open question below)
  created_at: "2026-09-12T10:00:00Z"
```

**Open question this raises:** `verifier_command` needs a home other than typing it on every invocation — most likely a per-project config file (e.g. `.council.yaml` in the repo root with a `verify:` key) that `council run` reads by default, with `--verify` as an override. That's the natural next thing to design once ingestion itself is settled, since it's the bridge into the verifier abstraction.

## 3. The Pass/Fail Verifier Abstraction

This is what decides PASS/FAIL for the escalation state machine (Section 4) and what `verifier_command` from Section 2 actually points at.

**Per-project config, not a single command.** A project's definition of "passing" is rarely one shell command — lint and test are usually separate concerns with different failure meanings. So `.council.yaml` defines an ordered list of steps:

```yaml
# .council.yaml, lives at the project root
verify:
  - name: "lint"
    run: "ruff check ."
    required: true
  - name: "test"
    run: "pytest tests/ -q"
    required: true
timeout_seconds: 120
working_dir: "."
```

- Steps run in order; a `required: true` step failing stops the run there (no point running tests if lint-breaking syntax errors exist).
- `required: false` steps can run and report without blocking — useful for things like a style check you want visibility on but don't want to gate escalation.

**Interface the classifier/escalation logic calls:**
```python
def run_verifier(config, workdir) -> VerifyReport:
    results = []
    for step in config.verify:
        try:
            proc = subprocess.run(
                step.run, cwd=workdir, shell=True, capture_output=True,
                timeout=step.timeout_seconds or config.timeout_seconds,
            )
            results.append(VerifyResult(step.name, passed=(proc.returncode == 0),
                                         stdout=proc.stdout, stderr=proc.stderr))
        except subprocess.TimeoutExpired:
            results.append(VerifyResult(step.name, passed=False, error="timeout"))
        if not results[-1].passed and step.required:
            break
    overall_pass = all(r.passed for r in results if r.required)
    return VerifyReport(results, overall_pass)
```

**Why capture stdout/stderr, not just a boolean.** The escalation model (whatever handles "debug root cause" on the second failure) needs the actual error output to do anything useful — a bare pass/fail tells it nothing. This report is what gets attached to the task when it escalates.

**A failure to distinguish: task failure vs. verifier failure.** If `pytest` itself isn't installed, or the sandbox has no network to fetch a dependency, that's an environment problem, not evidence the model's code is wrong — treating it as a normal FAIL would wrongly count against the model and trigger escalation for the wrong reason. Worth a distinct `VERIFY_ERROR` state that surfaces to the user directly ("your verify command doesn't run") instead of feeding into the retry/escalation counters at all.

**Not decided yet, deliberately:** whether v1 auto-detects a verify command when no `.council.yaml` exists (e.g. `package.json` → `npm test`, `Cargo.toml` → `cargo test`, `pyproject.toml` → `pytest`). Convenient, but a wrong guess silently verifying the wrong thing is worse than forcing an explicit config on first run. Leaning toward: require the config file in v1, add auto-detection later once you've seen how often people actually forget to write one.

## 4. Sandbox Scope

**v1 recommendation: plain Docker, no gVisor.** Here's the actual trade-off, not just a pick:

- **No isolation at all** isn't a real option here — this tool runs model-generated code and shell commands (the verify steps from Section 3) with real filesystem access on your machine. A hallucinated `rm -rf`, a bad `npm install` postinstall script, or a prompt-injected instruction from a file the model reads are all real risks once you're auto-executing AI output. Skipping sandboxing to save setup time trades a one-time cost for an open-ended one.
- **Docker + gVisor** (the original doc's proposal) gives the strongest isolation — gVisor intercepts syscalls so even a container escape has less to work with. But gVisor is Linux-only and doesn't integrate cleanly with Docker Desktop on macOS/Windows (which run containers inside a Linux VM already, and layering gVisor on top adds real friction). For an open-source tool you want broadly adoptable, that's a real tax on every non-Linux contributor before they've even run their first task.
- **Plain Docker** (no gVisor) — standard container isolation: separate filesystem namespace, can disable networking, resource limits (memory/CPU caps). Weaker than gVisor against a truly adversarial escape attempt, but this isn't a multi-tenant server running arbitrary strangers' code — it's a single user running their own council against their own project. Docker's isolation is proportionate to that threat model, and it's supported everywhere Docker Desktop runs.

```yaml
# .council.yaml sandbox block
sandbox:
  runtime: "docker"        # v1: docker only. "none" prints a loud warning, dev-use only.
  network: false           # default off; a step can request it explicitly (e.g. npm install)
  mount: "workdir:rw"      # only the target project directory — nothing else from the host
  resource_limits:
    memory: "2g"
    cpus: 2
  image: "python:3.11-slim"   # explicit per project — see note below
```

**Deliberately not auto-guessed:** the container image, same reasoning as the verifier command in Section 3 — auto-detecting "this looks like a Python project, use `python:3.11-slim`" is convenient but a wrong guess silently runs the wrong toolchain. v1 requires the user to set `image` explicitly; auto-detection is a later convenience layer once real usage shows what people actually need.

**gVisor gets deferred to an opt-in hardening flag**, not removed from the design — someone running this against untrusted third-party code (not their own project) should be able to turn it on. It just isn't the v1 default given the platform friction it adds for the common case.

## 5. API Key / Secrets Convention

**Keys live in environment variables only — never in the registry file, never in `.council.yaml`, never in task text.**

The model registry (Section 1) already points at this: each entry has `api_key_env: "OPENROUTER_API_KEY"` — a *name*, not a value. That's exactly what lets `models.yaml` be shared and committed in an open-source repo without exposing anything; the actual secret has to live somewhere that never gets committed.

**Where the value comes from:**
```
~/.config/council/.env      # global — applies across every project
./.env                      # project-local — overrides the global one if both are set
```
A committed `.env.example` (in the tool's own repo) lists every env var name the default registry expects, with no real values — so a new user knows exactly what to set without guessing or reverse-engineering it from the code.

```
# .env.example
OPENROUTER_API_KEY=
GROQ_API_KEY=
GOOGLE_AI_STUDIO_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

**Fail fast, not silent.** Before running any task, the orchestrator checks that every model in the active registry that could plausibly be selected has its `api_key_env` actually set. Missing → a clear error naming the exact variable, before any HTTP call goes out with a blank key.

**The sandbox never sees the keys — at all.** Worth stating explicitly, because it's easy to get wrong by default: the container from Section 4 only ever receives the project code and the verify command. It has no reason to hold an API key, since LLM calls are made by the host orchestrator process, not from inside the sandbox. If a key were passed into the container, a model-generated test step — or a compromised dependency the model pulled in — could read it straight out of the environment. Keeping that boundary hard (orchestrator holds secrets and talks to providers; sandbox only ever runs code and reports pass/fail) closes a real leak path that's easy to miss if it isn't a deliberate design decision.

**One key per provider, not per model.** `api_key_env` is shared across every registry entry from the same provider — every OpenRouter-routed model references `OPENROUTER_API_KEY`. You don't need five separate keys just because you've listed five OpenRouter models.

## 6. Task Classifier Logic

The classifier's job: given an incoming task, output `(complexity_tier, required_capabilities)`.

### Step 1 — Signal extraction
Pull cheap, local signals before calling any model:
- File/diff size (lines touched)
- Number of files touched (cross-file = higher complexity)
- Keyword match against a small rule set (`"security"`, `"auth"`, `"CVE"`, `"encrypt"` → force `security_audit` capability requirement)
- Whether it's a first attempt or a retry (retries escalate by default — see Step 3)
- Presence of a compile/test step already failing (structural signal, not guesswork)

### Step 2 — Complexity scoring (cheap, deterministic first)
```python
def classify_complexity(task):
    score = 0
    score += min(task.files_touched, 10) * 2
    score += min(task.lines_changed // 20, 10)
    score += 15 if task.cross_file_dependencies else 0
    score += 20 if task.is_retry else 0
    score += 25 if any(k in task.text.lower() for k in SECURITY_KEYWORDS) else 0

    if score < 20:
        return "simple"
    elif score < 50:
        return "moderate"
    else:
        return "complex"
```
This is intentionally boring and rule-based — no LLM call needed to decide "is this boilerplate," which is the whole point of the token-saving design. Only ambiguous cases (score near a threshold) should optionally get a cheap free-model classification pass instead of a hardcoded guess.

### Step 3 — Model selection
```python
def select_model(registry, complexity, required_caps, retry_count):
    candidates = [
        m for m in registry
        if all(c in m.capabilities for c in required_caps)
        and not any(c in m.not_trusted_for for c in required_caps)
    ]

    if complexity == "simple" and retry_count == 0:
        pool = [m for m in candidates if m.cost_tier == "free"]
    else:
        pool = candidates  # complex or already-retried tasks can see paid tier

    if not pool:
        raise NoEligibleModelError(required_caps)

    # within the eligible pool, prefer highest quality_score,
    # tie-break on lowest cost, then lowest latency_class
    return max(pool, key=lambda m: (m.quality_score, -m.cost_per_1k_input))
```

### Step 4 — Escalation state machine
Track per-task state, not per-model state:

```
NEW → (assign free model) → RUNNING
RUNNING → PASS → DONE
RUNNING → FAIL (attempt 1) → RUNNING (same tier, escalates_to fallback if repeat model fails)
RUNNING → FAIL (attempt 2) → ESCALATED (force paid tier regardless of complexity score)
ESCALATED → FAIL → HUMAN_REVIEW (never loop paid model indefinitely)
```

The `HUMAN_REVIEW` terminal state matters — without it, a paid model failing twice just burns tokens silently.

## 7. Human Review Handoff (the `HUMAN_REVIEW` terminal state)

This is what actually happens when the escalation chain from Section 6 runs out: free model fails, retries/escalates to paid, paid also fails. Rather than looping indefinitely or failing silently, it lands here.

**Nothing gets auto-applied.** The paid model's last attempted diff never touches the user's actual working tree — it stays exactly what it is, a rejected attempt, until a human looks at it.

**A review record gets written to disk, not a chat prompt:**
```
.council/reviews/<task-id>/
  summary.md          # task description, models tried in order, one plain-language
                       # paragraph on why it stopped — not a debug dump
  attempt-1.diff       # the actual proposed change from each attempt, preserved
  attempt-2.diff
  verify-log-1.txt     # captured stdout/stderr per attempt, from Section 3's verifier
  verify-log-2.txt
```

**The CLI exits non-zero and points at the folder** — it doesn't block waiting for someone at a keyboard. That matters if this ever runs inside a script or CI pipeline where no one's watching in real time; a blocking interactive prompt would just hang. A pointer plus a non-zero exit code is something both a human and a script can act on.

**`VERIFY_ERROR` from Section 3 lands in the same folder structure, tagged distinctly** (`environment_error: true` in `summary.md`) rather than looking like a normal model failure — a human still needs to see it, but the summary should say "the setup itself couldn't run" rather than implying the model's code was bad. That distinction is what keeps environment problems from wrongly counting against a model's future routing eligibility.

**Deliberately deferred:** auto-filing a GitHub issue, Slack/email notification. Both are convenience layers on top of "write a folder, exit non-zero" — not needed to have a working, honest terminal state in v1.

**Open question:** whether review folders get pruned automatically after some age/count, or just accumulate. Leaning toward letting them accumulate by default — losing debugging history to an auto-cleanup is a worse failure mode than a slowly growing `.council/reviews/` directory the user can prune themselves.

## 8. Setting `quality_score` — manual only (v1)

No scraper, no LLM-judge harness. Keep the automated core dumb and deterministic; scoring is either left blank, set by the user, or earned through real use:

```yaml
quality_score:
  value: null                  # null until scored
  source: null                 # user_override | measured
  last_updated: null
```

- **New model, no history** → `quality_score: null`. The classifier treats `null` as eligible-but-unranked: it can still be picked to fill a capability gap, but it never wins a tie-break against a model with a real score.
- **User override** → the user sets a number from their own experience whenever they've actually used the model ("this one's been solid for me on refactors" → `0.8`). Tagged `source: user_override`. No fetch job, no judge call, no maintenance burden.
- **`measured`** → once the council has run real tasks through a model and logged your own compile/test pass rate, that becomes the score, automatically, as a side effect of normal use — not extra infrastructure, just logging what already happens.

This trades early-model guessing (a brand-new model sits unranked until someone tries it) for zero scraper/judge machinery to build or trust. For a tool where every user's model mix differs, a generic public ranking wasn't going to be that reliable per-user anyway.

## 9. Open questions before this is buildable
- Does the registry live as a flat YAML file, or does it need to be a small local SQLite table once `quality_score` starts updating from real task history? Flat file is fine for v1; don't over-build this early.
- Who/what verifies `last_verified`? A cron job that pings each endpoint with a trivial prompt is a cheap way to catch dead free-tier models before a real task hits them.
- How does a `null`-scored model get picked among several unranked options for the same capability? A simple default (round-robin, or lowest cost first) is enough for v1 — don't build a second scoring system just to break ties among unranked models.

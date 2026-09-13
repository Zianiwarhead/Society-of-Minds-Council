"""Model executor — OpenRouter HTTPS client + OpenCode CLI backend.

Two backends, routed per model (`backend: openrouter | opencode`):
- openrouter: one HTTPS endpoint covers most free + paid models.
- opencode: shells out to `opencode run -m <provider/model>`. Auth is
  opencode's own business (its auth.json/env); the council never sees
  those keys. Any model opencode knows becomes a council mind.

Keys come from the environment via `model.api_key_env` — never logged,
never written to disk, never passed into the verifier sandbox.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


class ExecutorError(Exception):
    """Raised when a model call can't be completed (missing key, HTTP
    error, timeout, malformed response). Never includes the key."""


@dataclass
class ExecutorResult:
    model_id: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    error: Optional[str] = None  # set when the call itself failed

    @property
    def ok(self) -> bool:
        return self.error is None

    def estimated_cost_usd(self, cost_per_1k_input: float,
                           cost_per_1k_output: float) -> float:
        return (self.prompt_tokens / 1000 * cost_per_1k_input
                + self.completion_tokens / 1000 * cost_per_1k_output)


def _api_key(model) -> str:
    key = os.environ.get(model.api_key_env)
    if not key:
        raise ExecutorError(
            f"model '{model.id}' needs env var {model.api_key_env} — "
            f"set it before running"
        )
    return key


def build_prompt(task_description: str, file_context: Dict[str, str]) -> str:
    """Assemble the bake-off prompt: task + target file contents + a
    unified-diff-only instruction so the harness can apply + verify."""
    parts = [
        f"TASK: {task_description}",
        "",
        "CONTEXT FILES (current contents):",
    ]
    for path, content in file_context.items():
        parts += [f"--- {path} ---", content[:12000], ""]
    parts += [
        "INSTRUCTIONS:",
        "Propose the change as a unified diff (git format, `--- a/...` / `+++ b/...`).",
        "Output ONLY the diff inside a ```diff fenced block, no explanation.",
        "If the task needs no change, output an empty diff block.",
    ]
    return "\n".join(parts)


def build_create_prompt(task_description: str, filename: str) -> str:
    """Assemble a greenfield prompt: the model writes a whole file, not a
    diff. Output contract is one fenced code block so the harness can save
    + verify it in isolation."""
    return "\n".join([
        f"TASK: {task_description}",
        "",
        f"Write a complete, runnable file named {filename}.",
        "Rules:",
        "- Use only the Python standard library unless the task says otherwise.",
        "- The file must parse and import cleanly (it will be checked).",
        "- Output ONLY the file contents inside a single fenced code block",
        "  (```python ... ```), no explanation before or after.",
    ])


def build_review_prompt(task_description: str, filename: str, peer_code: str,
                        author_id: str) -> str:
    """Ask one model to REVIEW a peer's code as numbered corrections.
    The author must answer each one in the revise round — correct it or
    justify keeping it — so reviews become corrections, not ambiance."""
    return "\n".join([
        f"TASK: {task_description}",
        "",
        f"REVIEW a peer's solution ({filename}, written by {author_id}).",
        "Reply ONLY with a numbered list of concrete corrections:",
        "1. <file line or function> — <what is wrong> — <how to fix it>",
        "2. ...",
        "Rules: be specific and actionable. If something is good, say so in",
        "one line at the end under KEEP:. No code blocks, no rewrite —",
        "corrections only, the author applies them.",
        "",
        f"--- {author_id}'s {filename} ---",
        peer_code[:20000],
    ])


def build_critique_prompt(task_description: str, filename: str, code: str,
                          verifier_notes: str = "",
                          human_notes: str = "",
                          peer_reviews: str = "") -> str:
    """Assemble the revise round: the model gets a peer's code plus what
    the verifier said, and returns a better whole file."""
    parts = [
        f"TASK: {task_description}",
        "",
        f"A peer model wrote this {filename}. Improve it — fix bugs,",
        "handle edge cases, keep what works. Return the COMPLETE improved",
        "file, not a diff.",
        "",
        f"--- current {filename} ---",
        code[:20000],
        "",
        "First, answer every numbered correction from the peer review,",
        "one per line: `FIXED: <n>` or `KEPT: <n> because <reason>`.",
        "Then output the full file. A correction you silently ignore is",
        "a bug you chose to keep.",
        "",
    ]
    if verifier_notes:
        parts += ["VERIFIER SAID:", verifier_notes[:2000], ""]
    if peer_reviews:
        parts += ["PEER REVIEW SAID:", peer_reviews[:3000], ""]
    if human_notes:
        parts += ["THE HUMAN PLAYTESTED IT AND SAID:",
                  human_notes[:2000],
                  "Address every point. ", ""]
    parts += [
        "Output ONLY the full file inside a single ```python fenced block,",
        "no explanation before or after.",
    ]
    return "\n".join(parts)


def extract_code_block(text: str) -> Optional[str]:
    """Pull the first fenced code block out of a response. Prefers a
    ```python block, falls back to any fenced block. Returns None if the
    model chatted instead of writing code."""
    lower = text.lower()
    for marker in ("```python", "```py", "```"):
        start = lower.find(marker)
        if start == -1:
            continue
        body_start = start + len(marker)
        end = text.find("```", body_start)
        if end == -1:
            continue
        code = text[body_start:end].strip("\n")
        if code.strip():
            return code
    return None


def extract_diff(text: str) -> Optional[str]:
    """Pull the first ```diff ... ``` block out of a response. Returns None
    if no fenced block exists (model chatted instead of diffing)."""
    lower = text.lower()
    start_markers = ["```diff", "```patch", "```"]
    for marker in start_markers:
        start = lower.find(marker)
        if start == -1:
            continue
        body_start = start + len(marker)
        end = text.find("```", body_start)
        if end == -1:
            continue
        diff = text[body_start:end].strip()
        if diff:
            return diff
    stripped = text.strip()
    if stripped.startswith("--- ") or stripped.startswith("diff --git"):
        return stripped
    return None


TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
RETRY_DELAY_SECONDS = 5


def _is_transient(exc: BaseException) -> bool:
    """Whether a failed call is worth one retry: dropped connections,
    timeouts, rate limits, and provider 5xx. Auth/validation errors and
    malformed payloads fail fast instead."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_CODES
    return isinstance(exc, (TimeoutError, OSError))


def call_model(model, prompt: str, timeout_seconds: int = 120,
               retries: int = 1,
               workdir: Optional[Path] = None) -> ExecutorResult:
    """Run one model call via the model's backend (openrouter HTTPS or
    `opencode run`). Transient failures retry once; anything definitive
    fails fast.
    """
    if getattr(model, "backend", "openrouter") == "opencode":
        return call_opencode(model, prompt, timeout_seconds, retries, workdir)
    return call_openrouter(model, prompt, timeout_seconds, retries)


def call_openrouter(model, prompt: str, timeout_seconds: int = 120,
                    retries: int = 1) -> ExecutorResult:
    """POST one chat completion. `model.endpoint` is the full
    chat-completions URL; `model.id` is the OpenRouter model id."""
    from council.registry import ModelEntry  # local import: keeps module import-light

    assert isinstance(model, ModelEntry)
    key = _api_key(model)
    body = json.dumps({
        "model": model.id,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        model.endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "society-of-minds-council/0.1",
            "HTTP-Referer": "https://github.com/society-of-minds-council",
            "X-Title": "Society of Minds Council",
        },
        method="POST",
    )
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                payload = json.load(resp)
            break
        except Exception as exc:  # noqa: BLE001 — classified below
            transient = _is_transient(exc)
            if transient and attempt < retries:
                attempt += 1
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode()[:500]
                except Exception:
                    detail = ""
                raise ExecutorError(
                    f"model '{model.id}' HTTP {exc.code}: {detail or exc.reason}"
                    + (f" (after {attempt + 1} tries)" if attempt else "")
                ) from exc
            if isinstance(exc, TimeoutError):
                raise ExecutorError(
                    f"model '{model.id}' timed out after {timeout_seconds}s"
                    + (f" (after {attempt + 1} tries)" if attempt else "")
                ) from exc
            if isinstance(exc, OSError):
                raise ExecutorError(
                    f"model '{model.id}' request failed: {exc}"
                    + (f" (after {attempt + 1} tries)" if attempt else "")
                ) from exc
            raise

    latency = time.monotonic() - started
    try:
        text = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        preview = json.dumps(payload)[:300]
        raise ExecutorError(
            f"model '{model.id}' returned an unexpected response shape: {preview}"
        ) from exc
    usage = payload.get("usage", {}) or {}
    return ExecutorResult(
        model_id=model.id,
        text=text,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_seconds=latency,
    )


def _collect_text_nodes(node, out: List[str]) -> None:
    """Best-effort harvest of assistant text from `opencode run --format
    json` events. Schema may drift across versions, so this gathers every
    string under text-ish keys instead of trusting one exact shape."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("text", "content", "message", "output", "result"):
                if isinstance(value, str) and value.strip():
                    out.append(value)
                else:
                    _collect_text_nodes(value, out)
            elif isinstance(value, (dict, list)):
                _collect_text_nodes(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_text_nodes(item, out)


def _parse_opencode_output(stdout: str) -> str:
    """Extract the agent's answer from `--format json` output. Falls back
    to raw stdout — fenced code blocks survive either way, which is all
    the harness strictly needs."""
    text = stdout.strip()
    if not text:
        return ""
    events: List[object] = []
    try:
        events.append(json.loads(text))
    except ValueError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    if not events:
        return text  # not JSON at all — treat the whole thing as the answer
    found: List[str] = []
    for event in events:
        _collect_text_nodes(event, found)
    # Drop echoes of our own prompt (some formats repeat the input).
    found = [t for t in found if len(t) > 20]
    return "\n\n".join(found).strip() or text


def call_opencode(model, prompt: str, timeout_seconds: int = 180,
                  retries: int = 1,
                  workdir: Optional[Path] = None) -> ExecutorResult:
    """Run one prompt through `opencode run -m <provider/model>`.

    For opencode-backend entries, `model.endpoint` holds the opencode
    model id (`provider/model`); auth is opencode's own business.
    """
    from council.registry import ModelEntry

    assert isinstance(model, ModelEntry)
    if shutil.which("opencode") is None:
        raise ExecutorError(
            f"model '{model.id}' needs the opencode CLI on PATH "
            f"(backend: opencode) — install it or drop this mind"
        )
    opencode_model = (model.endpoint or "").strip()
    if not opencode_model or opencode_model.startswith("http"):
        raise ExecutorError(
            f"model '{model.id}' has backend opencode but no opencode model id "
            f"in `endpoint` (want `provider/model`, e.g. anthropic/claude-sonnet-4-5)"
        )
    cmd = ["opencode", "run", "--model", opencode_model, "--format", "json", prompt]
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            break
        except subprocess.TimeoutExpired as exc:
            if attempt < retries:
                attempt += 1
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise ExecutorError(
                f"model '{model.id}' (opencode) timed out after {timeout_seconds}s"
            ) from exc
        except OSError as exc:
            if attempt < retries:
                attempt += 1
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise ExecutorError(
                f"model '{model.id}' (opencode) failed to launch: {exc}") from exc
    latency = time.monotonic() - started
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "unknown error").strip()[:500]
        raise ExecutorError(f"model '{model.id}' (opencode) exited {proc.returncode}: {err}")
    text = _parse_opencode_output(proc.stdout or "")
    if not text:
        raise ExecutorError(
            f"model '{model.id}' (opencode) returned no usable text")
    return ExecutorResult(model_id=model.id, text=text, latency_seconds=latency)


def safe_call(model, prompt: str, timeout_seconds: int = 120,
              workdir: Optional[Path] = None) -> ExecutorResult:
    """Like `call_model` but returns a failed result instead of raising —
    the compare harness must record per-model failures, not abort."""
    started = time.monotonic()
    try:
        return call_model(model, prompt, timeout_seconds, workdir=workdir)
    except ExecutorError as exc:
        return ExecutorResult(model_id=model.id, text="", error=str(exc),
                              latency_seconds=time.monotonic() - started)

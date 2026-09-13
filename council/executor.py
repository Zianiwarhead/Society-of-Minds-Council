"""Model executor — one OpenRouter Chat Completions client (stdlib only).

v1 scope: OpenRouter only. One HTTPS endpoint covers most free + paid
models, so a single client unblocks the bake-off without a per-provider
zoo. Other providers (direct Anthropic/OpenAI, local) stay future work.

Keys come from the environment via `model.api_key_env` — never logged,
never written to disk, never passed into the verifier sandbox.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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


def build_critique_prompt(task_description: str, filename: str, code: str,
                          verifier_notes: str = "") -> str:
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
    ]
    if verifier_notes:
        parts += ["VERIFIER SAID:", verifier_notes[:2000], ""]
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


def call_model(model, prompt: str, timeout_seconds: int = 120) -> ExecutorResult:
    """POST one chat completion via the model's endpoint. `model.endpoint`
    is the full chat-completions URL; `model.id` is the OpenRouter model id."""
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
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode()[:500]
        except Exception:
            detail = ""
        raise ExecutorError(
            f"model '{model.id}' HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise ExecutorError(
            f"model '{model.id}' timed out after {timeout_seconds}s") from exc
    except OSError as exc:
        raise ExecutorError(f"model '{model.id}' request failed: {exc}") from exc

    latency = time.monotonic() - started
    try:
        text = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ExecutorError(
            f"model '{model.id}' returned an unexpected response shape") from exc
    usage = payload.get("usage", {}) or {}
    return ExecutorResult(
        model_id=model.id,
        text=text,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_seconds=latency,
    )


def safe_call(model, prompt: str, timeout_seconds: int = 120) -> ExecutorResult:
    """Like `call_model` but returns a failed result instead of raising —
    the compare harness must record per-model failures, not abort."""
    try:
        return call_model(model, prompt, timeout_seconds)
    except ExecutorError as exc:
        return ExecutorResult(model_id=model.id, text="", error=str(exc))

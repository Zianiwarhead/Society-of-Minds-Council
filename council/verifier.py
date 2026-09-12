"""Verifier runner — executes the ordered `verify` steps from `.council.yaml`
and reports what happened. See spec Section 3 (Pass/Fail Verifier Abstraction).

This is what turns "the model wrote some code" into the PASS/FAIL signal
the classifier's escalation state machine (Section 6) actually consumes —
and it's the one place responsible for telling a real merit-based failure
apart from the verifier itself being unable to run (Section 3/7's
VERIFY_ERROR distinction).
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from council.config import CouncilConfig, VerifyStep


def _executable_missing(command: str) -> bool:
    """True if the command's executable can't be resolved on PATH.

    Portable complement to the 126/127 exit-code check: Windows shells
    return 1 (not 127) for an unknown command, so the exit code alone
    can't distinguish "test failed" from "test runner isn't installed".
    """
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not parts:
        return False
    return shutil.which(parts[0]) is None


@dataclass
class StepResult:
    name: str
    passed: bool
    required: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: Optional[str] = None  # set when the step itself couldn't be run at all


@dataclass
class VerifyReport:
    steps: List[StepResult] = field(default_factory=list)
    overall_pass: bool = False
    verify_error: bool = False   # True: environment problem, not a merit-based failure

    def failing_step(self) -> Optional[StepResult]:
        """The first required step that failed, if any — what an escalation
        model would actually need to see to debug the root cause."""
        for step in self.steps:
            if step.required and not step.passed:
                return step
        return None


def run_verifier(config: CouncilConfig, workdir: Path) -> VerifyReport:
    """Runs each configured step, in order, stopping at the first required
    failure. Distinguishes a real failing step (command ran, exited nonzero)
    from a step that couldn't run at all (command not found, timeout) — the
    latter is flagged as `verify_error` and must never count as evidence the
    model's code was wrong (see Section 7)."""
    results: List[StepResult] = []
    verify_error = False

    for step in config.verify:
        timeout = step.timeout_seconds or config.timeout_seconds
        try:
            proc = subprocess.run(
                step.run,
                cwd=workdir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result = StepResult(
                name=step.name,
                passed=(proc.returncode == 0),
                required=step.required,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            result = StepResult(
                name=step.name,
                passed=False,
                required=step.required,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
                error=f"step timed out after {timeout}s",
            )
            if step.required:
                verify_error = True
        except OSError as exc:
            # The shell itself couldn't even start the command — an
            # environment problem, not the model's fault.
            result = StepResult(
                name=step.name, passed=False, required=step.required, error=str(exc)
            )
            if step.required:
                verify_error = True
        else:
            # shell=True routes "command not found" through the shell rather
            # than raising OSError — on Unix it comes back as exit 126/127,
            # but on Windows cmd.exe returns 1 with "not recognized" on
            # stderr. Either way it's an environment problem, not evidence
            # the model's code is wrong, so check the executable directly
            # instead of trusting the exit code alone.
            if proc.returncode in (126, 127) or (
                proc.returncode != 0 and _executable_missing(step.run)
            ):
                result.error = f"command not found or not executable (exit {proc.returncode})"
                if step.required:
                    verify_error = True

        results.append(result)

        if not result.passed and step.required:
            break

    overall_pass = all(r.passed for r in results if r.required)
    return VerifyReport(steps=results, overall_pass=overall_pass, verify_error=verify_error)

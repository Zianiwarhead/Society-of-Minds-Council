"""Boot-check verifier for tkinter (or any GUI) bake-offs.

`py_compile` only proves a file parses — cohere's 509-line council game
parsed fine and crashed on launch (AttributeError in __init__). This
check launches game.py as a subprocess instead:

- Still running after BOOT_WAIT seconds  -> it boots: PASS
- Exited early with a traceback/nonzero  -> it crashes: FAIL

Usage in .council.yaml:
  verify:
    - name: "boots"
      run: "python verify_game.py"
      required: true
"""
from __future__ import annotations

import subprocess
import sys

GAME_FILE = "game.py"
BOOT_WAIT_SECONDS = 8


def main() -> int:
    try:
        proc = subprocess.Popen(
            [sys.executable, GAME_FILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        print(f"could not launch {GAME_FILE}: {exc}")
        return 2  # environment problem, not a game bug

    try:
        out, _ = proc.communicate(timeout=BOOT_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        # Still alive after the wait: it boots and runs. Kill the window.
        proc.kill()
        proc.communicate()
        print(f"{GAME_FILE} booted and ran for {BOOT_WAIT_SECONDS}s: PASS")
        return 0

    # Exited on its own before the wait ran out.
    out = out or ""
    if proc.returncode == 0 and "Traceback" not in out:
        print(f"{GAME_FILE} exited cleanly: PASS")
        return 0
    print(f"{GAME_FILE} crashed on boot: FAIL")
    print(out[-2000:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

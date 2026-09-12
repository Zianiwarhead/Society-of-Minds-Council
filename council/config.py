"""Loads .council.yaml — verify steps (Section 3) and sandbox settings
(Section 4). Deliberately does not auto-detect either one: a wrong guess
silently running the wrong toolchain is worse than requiring an explicit
config on first run."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class VerifyStep:
    name: str
    run: str
    required: bool = True
    timeout_seconds: Optional[int] = None


@dataclass
class SandboxConfig:
    runtime: str = "docker"
    network: bool = False
    mount: str = "workdir:rw"
    image: Optional[str] = None
    memory: str = "2g"
    cpus: int = 2


@dataclass
class CouncilConfig:
    verify: list
    timeout_seconds: int = 120
    working_dir: str = "."
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


class ConfigError(Exception):
    """Raised when .council.yaml is missing or malformed in a way v1 won't
    guess around."""


def load_config(project_dir: Path) -> CouncilConfig:
    config_path = project_dir / ".council.yaml"
    if not config_path.exists():
        raise ConfigError(
            f"no .council.yaml found in {project_dir}. Create one with at "
            f"least a 'verify' step and a sandbox 'image' — see the spec "
            f"for the format."
        )

    raw = yaml.safe_load(config_path.read_text()) or {}

    verify_steps = [
        VerifyStep(
            name=step["name"],
            run=step["run"],
            required=step.get("required", True),
            timeout_seconds=step.get("timeout_seconds"),
        )
        for step in raw.get("verify", [])
    ]
    if not verify_steps:
        raise ConfigError(".council.yaml has no 'verify' steps defined")

    sandbox_raw = raw.get("sandbox", {})
    resource_limits = sandbox_raw.get("resource_limits", {})
    sandbox = SandboxConfig(
        runtime=sandbox_raw.get("runtime", "docker"),
        network=sandbox_raw.get("network", False),
        mount=sandbox_raw.get("mount", "workdir:rw"),
        image=sandbox_raw.get("image"),
        memory=resource_limits.get("memory", "2g"),
        cpus=resource_limits.get("cpus", 2),
    )
    if sandbox.runtime == "docker" and not sandbox.image:
        raise ConfigError(
            "sandbox.image is required in .council.yaml — v1 does not "
            "guess a container image from the project type (see Section 4)"
        )

    return CouncilConfig(
        verify=verify_steps,
        timeout_seconds=raw.get("timeout_seconds", 120),
        working_dir=raw.get("working_dir", "."),
        sandbox=sandbox,
    )

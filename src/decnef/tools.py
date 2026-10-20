"""External command-line tools (dcm2niix, ANTs, FSL, FreeSurfer, AFNI).

`run(cmd)` executes a command list and raises `ToolError` with the captured
output on failure. `require(name)` checks a tool is on PATH up front so a run
fails at start, not at volume 1. No conda activation, no shell strings.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger(__name__)


class ToolError(RuntimeError):
    pass


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def require(*names: str) -> dict[str, str]:
    """Resolve each tool on PATH; raise ToolError naming every missing one."""
    found, missing = {}, []
    for name in names:
        path = which(name)
        if path is None:
            missing.append(name)
        else:
            found[name] = path
    if missing:
        raise ToolError(f"required tool(s) not found on PATH: {', '.join(missing)}")
    return found


def run(cmd: Sequence[str], env: Optional[dict] = None, cwd: Optional[Path] = None, log_output: bool = False) -> subprocess.CompletedProcess:
    """Run `cmd` (a list), capture stdout+stderr; raise ToolError on non-zero exit."""
    cmd = [str(c) for c in cmd]
    log.debug("run: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd)
    if log_output and result.stdout:
        log.info(result.stdout)
    if result.returncode != 0:
        raise ToolError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}")
    return result


def env_with(**extra: str) -> dict:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in extra.items()})
    return env

"""
Utility functions for the LBWSG PAF simulation workflow.

Ported from vivarium_gates_mncnh's tools/utilities.py.
"""

import importlib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Literal, Optional, TextIO, Tuple

import pandas as pd

#: The two environment flavours this repository builds. Their dependency sets
#: conflict and cannot be merged into one environment.
EnvType = Literal["simulation", "artifact"]
ENV_TYPES: Tuple[EnvType, ...] = ("simulation", "artifact")


def _environment_sh_hint(env_type: Optional[EnvType] = None) -> str:
    """Return the ``source environment.sh`` invocation that builds *env_type*."""
    flags = "-s" if env_type in (None, "simulation") else f"-s -t {env_type}"
    return f"source environment.sh {flags}"


def _env_satisfies(env_type: EnvType) -> bool:
    """Return whether the *active* interpreter can do *env_type* work.

    Decided by whether ``vivarium_inputs`` is importable: it is installed in the
    artifact environment and not in the simulation one.
    """
    try:
        importlib.import_module("vivarium_inputs")
        artifact_capable = True
    except ImportError:
        artifact_capable = False

    if env_type == "artifact":
        return artifact_capable
    if env_type == "simulation":
        return shutil.which("psimulate") is not None and not artifact_capable
    raise ValueError(f"Unknown environment type '{env_type}'. Expected one of {ENV_TYPES}.")


def require_environment(env_type: EnvType) -> None:
    """Abort unless the *active* environment can do *env_type* work.

    Parameters
    ----------
    env_type
        The flavour of work about to be done.

    Raises
    ------
    ValueError
        If *env_type* is not one of :data:`ENV_TYPES`.
    RuntimeError
        If the active environment cannot do that work.
    """
    if env_type not in ENV_TYPES:
        raise ValueError(
            f"Unknown environment type '{env_type}'. Expected one of {ENV_TYPES}."
        )

    if _env_satisfies(env_type):
        print(f"  \u2713 active environment ({sys.prefix}) can do {env_type} work")
        return

    other = "artifact" if env_type == "simulation" else "simulation"
    raise RuntimeError(
        f"This is not a {env_type} environment: {sys.prefix}\n"
        f"  It looks like a {other} environment instead.\n"
        f"  Build or activate the right one with '{_environment_sh_hint(env_type)}'.\n"
        "  If this is a workflow step, set its 'environment' key to the "
        f"{env_type} environment."
    )


def _feed_confirmations(stream: TextIO) -> None:
    """Answer ``y`` to every prompt the child writes, until it closes stdin."""
    # Answering once would leave a command that prompts twice blocked forever.
    try:
        while True:
            stream.write("y\n")
            stream.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (BrokenPipeError, OSError, ValueError):
            pass


def run_command(
    cmd: List[str],
    description: str,
    auto_confirm: bool = False,
    capture_full_output: bool = False,
) -> Optional[str]:
    """
    Run a command in the active environment and handle errors.

    Parameters
    ----------
    cmd
        Command and arguments to execute
    description
        Description of what the command does (for error messages)
    auto_confirm
        If True, answer 'y' to any prompts (useful for make_artifacts)
    capture_full_output
        If True, capture and return the full command output as a string

    Returns
    -------
        The full output if capture_full_output is True, otherwise None

    Raises
    ------
    RuntimeError
        If the executable is not found, or if the command exits non-zero.
    """
    argv = list(cmd)

    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Environment: {sys.prefix}")
    print(f"Command: {' '.join(argv)}")
    if auto_confirm:
        print("Auto-confirm: y (continuous)")
    print(f"{'='*80}\n")

    try:
        # Start in a new process group so an interrupt can kill the whole tree.
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if auto_confirm else None,
            stdout=subprocess.PIPE if capture_full_output else None,
            stderr=subprocess.STDOUT if capture_full_output else None,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Failed {description}: '{argv[0]}' was not found.\n"
            f"Build the environment that provides it with "
            f"'{_environment_sh_hint()}' (add '-t artifact' for artifact tooling)."
        )

    if auto_confirm:
        confirm_thread = threading.Thread(
            target=_feed_confirmations, args=(process.stdin,), daemon=True
        )
        confirm_thread.start()

    full_output: List[str] = []
    try:
        if capture_full_output:
            for line in process.stdout:
                print(line, end="")
                full_output.append(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        print(f"\nInterrupted. Terminating {description}...")
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()
        raise

    if return_code != 0:
        raise RuntimeError(f"Failed {description}. Exit code: {return_code}")

    return "".join(full_output) if capture_full_output else None


def check_psimulate_finished(psimulate_output: str) -> bool:
    """Return whether psimulate reports that every job completed successfully.

    The status is the last "(M of N total jobs completed successfully overall)"
    line in the output.
    """
    print("\nChecking psimulate completion status...")
    match = _last_match(
        psimulate_output, r"\((\d+) of (\d+) total jobs completed successfully overall\)"
    )
    if match is None:
        print("WARNING: Could not find the completion status in the psimulate output")
        return False

    completed, total = int(match.group(1)), int(match.group(2))
    print(f"Job completion: {completed} of {total}")
    if completed == total:
        print("✓ All jobs completed successfully")
        return True
    print(f"✗ Only {completed} of {total} jobs completed successfully")
    return False


def extract_results_dir(psimulate_output: str) -> Optional[str]:
    """Return the results directory from the last "Results written to:" line, if any."""
    match = _last_match(psimulate_output, r"Results written to:\s*(.+)")
    if match is None:
        print("WARNING: Could not find the results directory in the psimulate output")
        return None
    # psimulate colors its log even when the output is piped.
    results_dir = re.sub(r"\x1b\[[0-9;]*m", "", match.group(1)).strip()
    print(f"Results directory: {results_dir}")
    return results_dir


def _last_match(output: str, pattern: str) -> Optional[re.Match]:
    """Return the match of pattern on the last output line that has one."""
    for line in reversed(output.splitlines()):
        match = re.search(pattern, line)
        if match:
            return match
    return None


def read_psimulate_results(results_dir: str, measure: str) -> pd.DataFrame:
    """Read every shard psimulate wrote for one measure into a single DataFrame.

    The shards are read, not moved, so the run directory keeps its results.
    """
    measure_dir = Path(results_dir) / measure
    shards = sorted(measure_dir.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"psimulate wrote no results for {measure} under {measure_dir}")
    return pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)

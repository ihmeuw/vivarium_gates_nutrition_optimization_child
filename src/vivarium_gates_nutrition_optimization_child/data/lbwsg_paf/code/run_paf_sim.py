#!/usr/bin/env python
"""
Run the whole LBWSG PAF workflow, in order.

Ported from vivarium_gates_mncnh's driver. The workflow alternates between the
artifact and simulation environments, whose dependencies conflict and cannot be
merged. ``run_paf_sim_step.py`` runs ONE phase per invocation, wholly inside
whichever environment is active, so something has to enter the right
environment between phases. This script is that something, both by hand and as
the ``lbwsg_pafs`` step of ``model_specifications/artifact_workflow.yaml``.

Usage
-----
Run it from the simulation environment; it enters the right one for each phase::

    python run_paf_sim.py -l Ethiopia -o /path/to/new/dir
    python run_paf_sim.py -l Ethiopia -o /path/to/new/dir --artifact-env my_artifact_env

Every other argument is passed through to ``run_paf_sim_step.py`` unchanged.
After fixing whatever made a phase fail, continue from it rather than starting
over::

    python run_paf_sim.py -l Ethiopia -o /path/to/new/dir --from enn-artifact

or run that one phase on its own, from an activated environment of the right
kind::

    python run_paf_sim_step.py --step enn-artifact -l Ethiopia -o /path/to/new/dir
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vivarium_gates_nutrition_optimization_child.constants.metadata import PAF_PHASES

try:
    from vivarium.cluster_tools.core.jobmon.env import (
        resolve_env_bin_path,
        resolve_env_prefix,
    )
except ImportError as e:  # pragma: no cover - depends on how the env was built
    raise SystemExit(
        f"Could not import the environment resolver from vivarium.cluster_tools ({e}).\n"
        "Run this script from the simulation environment. The artifact "
        "environment's jobmon cannot be imported where setuptools no longer "
        "provides pkg_resources."
    )

RUN_PAF_SIM_STEP = Path(__file__).resolve().parent / "run_paf_sim_step.py"

#: Names the child should not inherit: they describe the environment this
#: script is running in, not the one each phase is launched into.
STALE_ENV_MARKERS = ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME")


def _repo_root() -> Path:
    """Return the repository root."""
    result = subprocess.run(
        ["git", "-C", str(RUN_PAF_SIM_STEP.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _dist_name(repo_root: Path) -> str:
    """Return the distribution name, from the single copy of it in pyproject.toml."""
    result = subprocess.run(
        ["make", "-s", "-C", str(repo_root), "print-dist-name"],
        capture_output=True,
        text=True,
        check=True,
    )
    name = result.stdout.strip()
    if not name:
        raise SystemExit("Could not determine the distribution name from pyproject.toml")
    return name


def _default_env(flavour: str, dist_name: str, repo_root: Path) -> str:
    """Return the environment to use for *flavour* when none was given.

    The overlay is named by path rather than by name because the resolver
    matches a bare name against ``.venv`` under the *current* directory, and
    this script should work from anywhere in the repository.
    """
    overlay = repo_root / ".venv" / f"{dist_name}_{flavour}"
    return str(overlay) if overlay.is_dir() else f"{dist_name}_{flavour}"


def _resolve_environments(args: argparse.Namespace) -> Dict[str, Tuple[str, str]]:
    """Resolve every flavour to a ``(prefix, PATH additions)`` pair, up front.

    Resolving all of them before running anything means a mistyped environment
    fails immediately rather than partway through the sequence, after earlier
    phases have already rebuilt the artifact.
    """
    # Finding these runs git and make, so only do it if we actually need a
    # default. When both environments are named, we never do.
    repo_root: Optional[Path] = None
    dist_name: Optional[str] = None

    resolved = {}
    for flavour in sorted({flavour for _, flavour in PAF_PHASES}):
        env_spec = getattr(args, f"{flavour}_env")
        if not env_spec:
            if repo_root is None:
                repo_root = _repo_root()
                dist_name = _dist_name(repo_root)
            env_spec = _default_env(flavour, str(dist_name), repo_root)
        try:
            prefix = resolve_env_prefix(env_spec)
        except RuntimeError as e:
            raise SystemExit(
                f"Could not use '{env_spec}' as the {flavour} environment.\n"
                f"  {e}\n"
                f"Name one with --{flavour}-env, or build this repository's "
                f"with 'source environment.sh -s -t {flavour}'."
            )
        resolved[flavour] = (prefix, resolve_env_bin_path(prefix))
        print(f"  {flavour:10} environment: {env_spec} -> {prefix}")
    return resolved


def _run_step(
    step: str, flavour: str, prefix: str, bin_path: str, passthrough: List[str]
) -> None:
    """Run one phase in the environment at *prefix*."""
    print()
    print("#" * 80)
    print(f"# step '{step}'  --  {flavour} environment: {prefix}")
    print("#" * 80)

    child_env = {k: v for k, v in os.environ.items() if k not in STALE_ENV_MARKERS}
    child_env["PATH"] = f"{bin_path}:{child_env['PATH']}"

    subprocess.run(
        [f"{prefix}/bin/python", str(RUN_PAF_SIM_STEP), "--step", step, *passthrough],
        env=child_env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run every phase of the LBWSG PAF workflow in order, entering the "
            "right environment for each. Unrecognised arguments are passed "
            "through to run_paf_sim_step.py."
        )
    )
    for flavour in sorted({flavour for _, flavour in PAF_PHASES}):
        parser.add_argument(
            f"--{flavour}-env",
            default=None,
            metavar="ENV",
            help=(
                f"The {flavour} environment: a conda environment name, a venv "
                "name, or a path to either's prefix. Defaults to this "
                "repository's overlay, or its conda environment."
            ),
        )
    parser.add_argument(
        "--from",
        dest="start_at",
        default=None,
        choices=[phase for phase, _ in PAF_PHASES],
        help=(
            "Start at this phase rather than the first, to continue a run after "
            "fixing whatever made a phase fail."
        ),
    )
    args, passthrough = parser.parse_known_args()

    if "--step" in passthrough:
        parser.error(
            "--step is chosen per phase by this script. To run a single phase, "
            "invoke run_paf_sim_step.py directly from an activated environment."
        )

    phases = list(PAF_PHASES)
    if args.start_at:
        skipped = [phase for phase, _ in phases].index(args.start_at)
        phases = phases[skipped:]

    print("\n" + "=" * 80)
    print(f"LBWSG PAF workflow -- {len(phases)} of {len(PAF_PHASES)} phases")
    if args.start_at:
        print(f"Starting at '{args.start_at}'; {skipped} earlier phase(s) skipped")
    print("=" * 80)
    environments = _resolve_environments(args)
    print("=" * 80)

    for step, flavour in phases:
        prefix, bin_path = environments[flavour]
        try:
            _run_step(step, flavour, prefix, bin_path, passthrough)
        except subprocess.CalledProcessError as e:
            print(
                f"\nStep '{step}' failed (exit code {e.returncode}). Earlier phases "
                "have already run; once you have fixed the cause, re-run this phase "
                f"on its own with:\n"
                f"  python {RUN_PAF_SIM_STEP} --step {step} {' '.join(passthrough)}\n"
                f"from the {flavour} environment. Or fix the cause and re-run "
                f"this script with --from {step} to continue from here.",
                file=sys.stderr,
            )
            raise SystemExit(e.returncode)

    print()
    print("#" * 80)
    print(f"# All {len(phases)} phases completed.")
    print("#" * 80)


if __name__ == "__main__":
    main()

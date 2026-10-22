#!/usr/bin/env python
"""
Run one phase of the LBWSG PAF workflow.

Adapted from vivarium_gates_mncnh's run_paf_sim_step.py.

Usage
-----
Run the five phases in order, each in the environment named beside it, or let
``run_paf_sim.py`` enter the environments for you::

    python run_paf_sim_step.py --step initial-artifact -l Ethiopia -o DIR   # artifact
    python run_paf_sim_step.py --step enn-paf          -l Ethiopia -o DIR   # simulation
    python run_paf_sim_step.py --step enn-artifact     -l Ethiopia -o DIR   # artifact
    python run_paf_sim_step.py --step lnn-paf          -l Ethiopia -o DIR   # simulation
    python run_paf_sim_step.py --step final-artifact   -l Ethiopia -o DIR   # artifact

DIR holds the PAF simulation's national artifact and each pass's results. Only
``lnn-paf`` writes the tracked output under data/lbwsg_paf/outputs, and only
after the late neonatal pass has passed its checks.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    metadata,
    paths,
)
from vivarium_gates_nutrition_optimization_child.data import lbwsg_paf_results
from vivarium_gates_nutrition_optimization_child.data.lbwsg_paf_checks import (
    check_location,
)
from vivarium_gates_nutrition_optimization_child.tools.utilities import (
    check_psimulate_finished,
    extract_results_dir,
    read_psimulate_results,
    require_environment,
    run_command,
)
from vivarium_gates_nutrition_optimization_child.utilities import sanitize_location

CODE_DIR = Path(__file__).resolve().parent
ENN_SPECIFICATION = CODE_DIR / "lbwsg_paf_enn.yaml"
LNN_SPECIFICATION = CODE_DIR / "lbwsg_paf.yaml"
BRANCHES = CODE_DIR / "lbwsg_paf_branches.yaml"


def _artifact_path(artifact_dir: Path, location: str) -> Path:
    """Return the PAF simulation artifact make_artifacts writes for a location."""
    return artifact_dir / f"{sanitize_location(location)}.hdf"


def _pass_results_path(artifact_dir: Path, location: str, pass_name: str) -> Path:
    """Return where a pass's concatenated results are kept beside the artifact."""
    return artifact_dir / f"{sanitize_location(location)}_{pass_name}_paf_results.parquet"


def _build_artifact(
    location: str, artifact_dir: Path, description: str, replace_paf: bool = False
) -> None:
    """Build the PAF simulation's national artifact, or replace its PAF key."""
    cmd = [
        "make_artifacts",
        "-vvv",
        "-l",
        location,
        "--for-lbwsg-pafs",
        "-o",
        str(artifact_dir),
    ]
    if replace_paf:
        cmd += ["-r", data_keys.LBWSG.PAF]
    run_command(cmd, description, auto_confirm=True)


def _check_paf_artifact(artifact_file: Path, require_complete: bool) -> None:
    """Refuse to use or change an artifact this workflow did not build."""
    # Imported here because the artifact phases and psimulate both have it.
    from vivarium.artifact import Artifact

    if not artifact_file.exists():
        raise SystemExit(f"{artifact_file} does not exist. Run the initial-artifact phase.")
    keys = set(Artifact(artifact_file).keys)
    expected = {key for group in data_keys.LBWSG_PAF_KEY_GROUPS for key in group}
    unexpected = sorted(key for key in keys - expected if not key.startswith("metadata."))
    if unexpected:
        raise SystemExit(
            f"{artifact_file} holds keys a PAF simulation artifact does not, such as "
            f"{unexpected[:3]}, so it is not one. Refusing to use it."
        )
    missing = sorted(expected - keys - {data_keys.LBWSG.PAF})
    if require_complete and missing:
        raise SystemExit(
            f"{artifact_file} is missing {missing}. The initial-artifact phase may have "
            "stopped partway; rebuild it with a new --output-dir."
        )


def _run_paf_simulation(
    location: str,
    artifact_file: Path,
    sim_results_dir: Path,
    model_specification: Path,
    description: str,
    queue: Optional[str],
) -> pd.DataFrame:
    """Run psimulate on one pass's specification and return its PAF results.

    The run directory is kept: its results are read, not moved.
    """
    output_dir = sim_results_dir / sanitize_location(location)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "psimulate",
        "run",
        "-vvv",
        "-P",
        metadata.CLUSTER_PROJECT,
        "-m",
        str(metadata.PAF_SIM_PEAK_MEMORY_GB),
        "-r",
        metadata.PAF_SIM_MAX_RUNTIME,
        "-i",
        str(artifact_file),
        "-o",
        str(output_dir),
    ]
    if queue:
        cmd += ["-q", queue]
    cmd += [str(model_specification), str(BRANCHES)]
    psimulate_output = run_command(cmd, description, capture_full_output=True)
    if not check_psimulate_finished(psimulate_output):
        raise RuntimeError(
            f"{description}: not all jobs finished successfully. After finishing the "
            "run with psimulate restart, rerun this phase with --psimulate-results-dir."
        )
    results_dir = extract_results_dir(psimulate_output)
    if results_dir is None:
        raise RuntimeError(
            f"{description}: psimulate reported success but its results directory "
            "could not be determined. Rerun this phase with --psimulate-results-dir."
        )
    return read_psimulate_results(results_dir, lbwsg_paf_results.LBWSG_PAF_MEASURE)


def _get_paf_results(
    location: str,
    artifact_file: Path,
    args: argparse.Namespace,
    model_specification: Path,
    description: str,
) -> pd.DataFrame:
    """Return a pass's PAF results, from an earlier psimulate run if one was named."""
    if args.psimulate_results_dir:
        print(f"Reading {description} results from {args.psimulate_results_dir}")
        return read_psimulate_results(
            args.psimulate_results_dir, lbwsg_paf_results.LBWSG_PAF_MEASURE
        )
    return _run_paf_simulation(
        location,
        artifact_file,
        Path(args.sim_results_dir).expanduser(),
        model_specification,
        description,
        args.queue,
    )


def _write_paf_to_artifact(artifact_file: Path, paf: pd.DataFrame) -> None:
    """Write the LBWSG PAF key into the PAF simulation artifact."""
    # Imported here because only the artifact phases need it.
    from vivarium.artifact import Artifact

    artifact = Artifact(artifact_file)
    if data_keys.LBWSG.PAF in artifact:
        artifact.replace(data_keys.LBWSG.PAF, paf)
    else:
        artifact.write(data_keys.LBWSG.PAF, paf)


def _run_one_location(
    step: str, location: str, artifact_dir: Path, args: argparse.Namespace
) -> None:
    """Run *step* for a single location."""
    artifact_file = _artifact_path(artifact_dir, location)
    enn_results_file = _pass_results_path(artifact_dir, location, "enn")
    lnn_results_file = _pass_results_path(artifact_dir, location, "lnn")

    if step == "initial-artifact":
        if artifact_file.exists():
            raise SystemExit(
                f"{artifact_file} already exists. Choose a new --output-dir, or continue "
                "an earlier run of this workflow with run_paf_sim.py --from enn-paf if "
                "that run's initial-artifact phase finished."
            )
        _build_artifact(location, artifact_dir, f"PAF simulation artifact for {location}")

    elif step == "enn-paf":
        _check_paf_artifact(artifact_file, require_complete=True)
        results = _get_paf_results(
            location,
            artifact_file,
            args,
            ENN_SPECIFICATION,
            f"early neonatal PAF simulation for {location}",
        )
        lbwsg_paf_results.format_lbwsg_paf(results, require_late_neonatal=False)
        lbwsg_paf_results.sort_paf_results(results).to_parquet(enn_results_file, index=False)
        print(f"Wrote early neonatal results to {enn_results_file}")

    elif step == "enn-artifact":
        _check_paf_artifact(artifact_file, require_complete=True)
        enn_paf = lbwsg_paf_results.format_lbwsg_paf(
            pd.read_parquet(enn_results_file), require_late_neonatal=False
        )
        _write_paf_to_artifact(artifact_file, enn_paf)
        print(f"Wrote the early neonatal PAF into {artifact_file}")

    elif step == "lnn-paf":
        _check_paf_artifact(artifact_file, require_complete=True)
        results = _get_paf_results(
            location,
            artifact_file,
            args,
            LNN_SPECIFICATION,
            f"late neonatal PAF simulation for {location}",
        )
        lbwsg_paf_results.check_early_neonatal_unchanged(
            pd.read_parquet(enn_results_file), results
        )
        lbwsg_paf_results.format_lbwsg_paf(results)
        results = lbwsg_paf_results.sort_paf_results(results)
        results.to_parquet(lnn_results_file, index=False)
        output_file = lbwsg_paf_results.get_output_path(location)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(output_file, index=False)
        print(f"Wrote the LBWSG PAF output to {output_file}")

    elif step == "final-artifact":
        _check_paf_artifact(artifact_file, require_complete=True)
        # The artifact environment may read a different checkout's tracked output.
        output_file = lbwsg_paf_results.get_output_path(location)
        if not pd.read_parquet(output_file).equals(pd.read_parquet(lnn_results_file)):
            raise SystemExit(
                f"{output_file} is not this run's late neonatal output {lnn_results_file}. "
                "Install both environments from the checkout lnn-paf wrote to."
            )
        _build_artifact(
            location,
            artifact_dir,
            f"LBWSG PAF in the PAF simulation artifact for {location}",
            replace_paf=True,
        )
        check_location(
            artifact_file, location, enn_results_file, paf_output_file=lnn_results_file
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one phase of the LBWSG PAF workflow.")
    parser.add_argument(
        "--step",
        required=True,
        # Ordered as constants.metadata.PAF_PHASES is, so --help lists the
        # phases in the order they run rather than alphabetically.
        choices=list(metadata.PAF_PHASE_ENVIRONMENTS),
        help="Which step to run. It runs in the environment the caller activated.",
    )
    parser.add_argument(
        "-l",
        "--location",
        action="append",
        dest="locations",
        required=True,
        metavar="LOCATION",
        help="Location to run, repeatable. Must be in constants.metadata.LOCATIONS.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help=(
            "Directory for the PAF simulation artifact and each pass's results. "
            "Use a new directory for each run."
        ),
    )
    parser.add_argument(
        "--sim-results-dir",
        default=str(paths.LBWSG_PAF_SIM_RESULTS_DIR),
        help="Directory under which psimulate writes its run directories.",
    )
    parser.add_argument(
        "--queue",
        default=None,
        help="Cluster queue for psimulate. By default psimulate picks one from the runtime.",
    )
    parser.add_argument(
        "--psimulate-results-dir",
        default=None,
        help=(
            "For enn-paf or lnn-paf with one location: read the results of an earlier "
            "psimulate run (its results directory) instead of launching a new one."
        ),
    )
    args = parser.parse_args()

    unknown = [location for location in args.locations if location not in metadata.LOCATIONS]
    if unknown:
        parser.error(
            f"Unknown location(s): {', '.join(unknown)}. "
            f"Expected some of {list(metadata.LOCATIONS)}."
        )
    if args.psimulate_results_dir and (
        args.step not in ("enn-paf", "lnn-paf") or len(args.locations) != 1
    ):
        parser.error(
            "--psimulate-results-dir applies to enn-paf or lnn-paf for one location."
        )
    artifact_dir = Path(args.output_dir).expanduser()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"LBWSG PAF workflow -- step '{args.step}'")
    print("=" * 80)
    print(f"Locations: {', '.join(args.locations)}")
    print(f"Artifact directory: {artifact_dir}")
    print("=" * 80)

    require_environment(metadata.PAF_PHASE_ENVIRONMENTS[args.step])

    for location in args.locations:
        _run_one_location(args.step, location, artifact_dir, args)

    print("\n" + "=" * 80)
    print(f"Step '{args.step}' completed for {len(args.locations)} location(s).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

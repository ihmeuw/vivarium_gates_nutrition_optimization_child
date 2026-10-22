#!/usr/bin/env python
"""
Check a location's LBWSG PAF against calculations made directly from its artifact.

``run_paf_sim.py`` runs these checks in its final phase. Run them on their own
against any artifact holding the PAF, for example a full model artifact::

    python check_paf_outputs.py -l Ethiopia --artifact /path/to/ethiopia.hdf \\
        --pregnancy-artifact /path/to/pregnancy/ethiopia.hdf
"""

import argparse
from pathlib import Path

from vivarium_gates_nutrition_optimization_child.data.lbwsg_paf_checks import (
    check_location,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a location's LBWSG PAF.")
    parser.add_argument("-l", "--location", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--enn-results",
        type=Path,
        default=None,
        help="The early neonatal pass's results, to check the artifact's ENN PAF against.",
    )
    parser.add_argument(
        "--pregnancy-artifact",
        type=Path,
        default=None,
        help="The pregnancy model's artifact, whose LBWSG exposure must equal the birth exposure.",
    )
    parser.add_argument(
        "--paf-output",
        type=Path,
        default=None,
        help="The PAF output to compare with. Defaults to the location's tracked output.",
    )
    args = parser.parse_args()
    check_location(
        args.artifact,
        args.location,
        enn_results_file=args.enn_results,
        pregnancy_artifact_file=args.pregnancy_artifact,
        paf_output_file=args.paf_output,
    )


if __name__ == "__main__":
    main()

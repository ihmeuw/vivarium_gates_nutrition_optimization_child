from typing import Tuple

import click
from loguru import logger
from vivarium.framework.utilities import handle_exceptions

from vivarium_gates_nutrition_optimization_child.constants import metadata, paths
from vivarium_gates_nutrition_optimization_child.tools import (
    build_artifacts,
    build_results,
    configure_logging_to_terminal,
)


@click.command()
@click.option(
    "-l",
    "--location",
    default="all",
    show_default=True,
    type=click.Choice(metadata.LOCATIONS + ["all"]),
    help=(
        "Location for which to make an artifact. Note: prefer building archives on the cluster.\n"
        'If you specify location "all" you must be on a cluster node.'
    ),
)
@click.option(
    "--national",
    "build_national",
    is_flag=True,
    help="Build artifacts at national level data instead of at the default subnational level.",
)
@click.option(
    "-o",
    "--output-dir",
    default=str(paths.ARTIFACT_ROOT),
    show_default=True,
    type=click.Path(),
    help="Specify an output directory. Directory must exist.",
)
@click.option(
    "-a", "--append", is_flag=True, help="Append to the artifact instead of overwriting."
)
@click.option("-r", "--replace-keys", multiple=True, help="Specify keys to overwrite")
@click.option("-v", "verbose", count=True, help="Configure logging verbosity.")
@click.option(
    "--pdb",
    "with_debugger",
    is_flag=True,
    help="Drop into python debugger if an error occurs.",
)
def make_artifacts(
    location: str,
    output_dir: str,
    append: bool,
    replace_keys: Tuple[str, ...],
    verbose: int,
    with_debugger: bool,
    build_national: bool,
) -> None:
    # This is a flag for national level data but we want default to be to write subnational data
    fetch_subnationals = not build_national
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(
        build_artifacts,
        logger,
        with_debugger=with_debugger,
    )
    main(
        location,
        output_dir,
        append or replace_keys,
        replace_keys,
        verbose,
        fetch_subnationals,
    )

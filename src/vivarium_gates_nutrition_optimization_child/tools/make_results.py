import shutil
from pathlib import Path

from loguru import logger

from vivarium_gates_nutrition_optimization_child.results_processing import (
    process_results,
)


def build_results(output_file: str, single_run: bool, disaggregate_seeds: bool) -> None:
    output_file = Path(output_file)
    measure_dir = output_file.parent / "count_data"
    # Get location for subnationals
    location = output_file.resolve().parent.parent.name.title()
    if measure_dir.exists():
        shutil.rmtree(measure_dir)
    measure_dir.mkdir(exist_ok=True, mode=0o775)

    logger.info(f"Reading in output data from {str(output_file)}.")
    data, keyspace = process_results.read_data(output_file, single_run)
    logger.info(f"Filtering incomplete data from outputs.")
    rows = len(data)
    data = process_results.filter_out_incomplete(data, keyspace)
    new_rows = len(data)
    logger.info(
        f"Filtered {rows - new_rows} from data due to incomplete information.  {new_rows} remaining."
    )
    if not disaggregate_seeds:
        data = process_results.aggregate_over_seed(data, location)
    logger.info(f"Computing raw count and proportion data.")
    measure_data = process_results.make_measure_data(data, disaggregate_seeds, location)
    logger.info(f"Writing raw count and proportion data to {str(measure_dir)}")
    measure_data.dump(measure_dir)
    logger.info("**DONE**")

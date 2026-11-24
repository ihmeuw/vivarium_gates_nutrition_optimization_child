import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytestmark = [
    pytest.mark.cluster,
    pytest.mark.slow,
    # TODO: remove this skip when main can build a new artifact (MIC-7861).
    pytest.mark.skip(
        reason="The model spec's artifact predates the neonatal wasting transition rate fix, "
        "and main cannot build a new artifact until the GBD 2023 data update."
    ),
]

# The run keeps the model spec's step size and is shortened to this many steps. The wasting
# exit rates from SAM are high enough that steps much longer than a week give exit
# probabilities that sum to more than 1, so the run cannot be shortened by lengthening the
# step. With 4-day steps, 50 steps (200 days) gives non-zero values in every result.
N_TIME_STEPS = 50


@pytest.fixture
def model_spec(tmp_path: Path) -> Path:
    model_spec_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "vivarium_gates_nutrition_optimization_child"
        / "model_specifications"
        / "nutrition_optimization_child.yaml"
    )
    with open(model_spec_path, "r") as file:
        ms = yaml.safe_load(file)

    # Shorten the run by moving the end date to N_TIME_STEPS steps after the start.
    start_date = pd.to_datetime(
        f'{ms["configuration"]["time"]["start"]["year"]}-'
        f'{ms["configuration"]["time"]["start"]["month"]}-'
        f'{ms["configuration"]["time"]["start"]["day"]}',
        format="%Y-%m-%d",
    )
    step_size = ms["configuration"]["time"]["step_size"]
    end_date = start_date + pd.Timedelta(days=N_TIME_STEPS * step_size)
    ms["configuration"]["time"]["end"] = {
        "year": end_date.year,
        "month": end_date.month,
        "day": end_date.day,
    }

    model_spec = tmp_path / "test_model_spec.yaml"
    with open(model_spec, "w") as file:
        yaml.dump(ms, file)

    return model_spec


# Checked as a subset so observers added to the spec do not break this test.
EXPECTED_RESULTS = [
    "deaths",
    "ylls",
    "ylds",
    "person_time_child_wasting",
    "transition_count_child_wasting",
    "person_time_child_stunting",
    "person_time_child_underweight",
    "person_time_diarrheal_diseases",
    "transition_count_diarrheal_diseases",
    "person_time_measles",
    "transition_count_measles",
    "person_time_lower_respiratory_infections",
    "transition_count_lower_respiratory_infections",
    "person_time_malaria",
    "transition_count_malaria",
]


def test_simulate_run(
    model_spec: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    with capsys.disabled():  # disabled so we can monitor job submissions
        print("\n\n*** RUNNING TEST ***\n")

        cmd = f"simulate run {str(model_spec)} -o {str(tmp_path)} -vvv"
        subprocess.run(
            cmd,
            shell=True,
            check=True,
        )
        with open(model_spec, "r") as file:
            model_spec = yaml.safe_load(file)
        location = Path(model_spec["configuration"]["input_data"]["artifact_path"]).stem

        results_files = list((tmp_path / location).rglob("*.parquet"))
        assert results_files
        assert set(EXPECTED_RESULTS) <= {file.stem for file in results_files}
        for file in results_files:
            df = pd.read_parquet(file)
            assert df["value"].notna().all()
            assert (df["value"] != 0.0).any()

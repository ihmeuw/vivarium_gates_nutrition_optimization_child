from pathlib import Path
import pickle
import sys

from loguru import logger
import numpy as np
import pandas as pd

from vivarium import Artifact, InteractiveContext
from vivarium_cluster_tools.utilities import mkdir
from vivarium_gbd_access import gbd

from vivarium_gates_nutrition_optimization_child.constants import data_keys, metadata


def get_relative_risks(config: Path, artifact_path: str, input_draw: int, random_seed: int, age_group_id: int) -> pd.DataFrame:
    logger.remove()
    sim = InteractiveContext(config, setup=False)
    sim.configuration.input_data.artifact_path = artifact_path
    artifact = Artifact(artifact_path)

    # Make mapper for age_group_ids
    gbd_age_bins = gbd.get_age_bins()
    name_to_id_mapper = dict(zip(gbd_age_bins.age_group_name, gbd_age_bins.age_group_id))

    age_bins = artifact.load(data_keys.POPULATION.AGE_BINS)
    age_group_ids = age_bins.reset_index()["age_group_name"].map(name_to_id_mapper)
    age_group_ids.index = age_bins.index

    age_bins["age_group_id"] = age_group_ids
    age_bins = age_bins.reset_index().set_index('age_group_id')

    age_start = age_bins.loc[age_group_id, 'age_start']
    age_end = age_bins.loc[age_group_id, 'age_end']

    year_start = 2019
    year_end = 2020

    sim.configuration.update({
        'input_data': {
            'input_draw_number': input_draw,
        },
        'randomness': {
            'random_seed': random_seed,
        },
        'population': {
            'age_start': age_start,
            'age_end': age_end
        }
    })
    sim.setup()

    pop = sim.get_population()
    gestational_ages = sim.get_value('gestational_age.birth_exposure')(pop.index)
    birth_weights = sim.get_value('birth_weight.birth_exposure')(pop.index)

    interpolators = artifact.load(data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR)

    def calculate_rr_by_sex(sex: str) -> float:
        sex_mask = pop['sex'] == sex
        row_index = (sex, age_start, age_end, year_start, year_end)
        interpolator = pickle.loads(bytes.fromhex(
            interpolators.loc[row_index, f'draw_{input_draw}']
        ))
        rrs = np.exp(interpolator(gestational_ages[sex_mask], birth_weights[sex_mask], grid=False))
        return rrs

    lbwsg_rrs = pd.DataFrame({'relative_risk': 1.0}, index=pop.index)
    lbwsg_rrs['sex'] = pop['sex']
    lbwsg_rrs.loc[lbwsg_rrs['sex'] == 'Female', 'relative_risk'] = calculate_rr_by_sex('Female')
    lbwsg_rrs.loc[lbwsg_rrs['sex'] == 'Male', 'relative_risk'] = calculate_rr_by_sex('Male')

    return lbwsg_rrs


def get_age_bin(config: Path, artifact_path: str, age_group_id: int) -> pd.Interval:
    sim = InteractiveContext(config, setup=False)
    sim.configuration.input_data.artifact_path = artifact_path
    artifact = Artifact(artifact_path)

    # Make mapper for age_group_ids
    gbd_age_bins = gbd.get_age_bins()
    name_to_id_mapper = dict(zip(gbd_age_bins.age_group_name, gbd_age_bins.age_group_id))

    age_bins = artifact.load(data_keys.POPULATION.AGE_BINS)
    age_group_ids = age_bins.reset_index()["age_group_name"].map(name_to_id_mapper)
    age_group_ids.index = age_bins.index

    age_bins["age_group_id"] = age_group_ids
    age_bins = age_bins.reset_index().set_index('age_group_id')
    age_bin = pd.Interval(age_bins.loc[age_group_id, 'age_start'], age_bins.loc[age_group_id, 'age_end'])

    return age_bin


def get_paf_for_age_group(config: Path, artifact_path: str, input_draw: int, random_seed: int, age_group_id: int) -> pd.DataFrame:
    age_bin = get_age_bin(config, artifact_path, age_group_id)
    relative_risks = pd.concat([get_relative_risks(config, artifact_path, input_draw, seed, age_group_id)
                                for seed in range(random_seed, random_seed + 10)])

    def calculate_paf_by_sex(sex: str) -> float:
        mean_rr = relative_risks.loc[relative_risks['sex'] == sex, 'relative_risk'].mean()
        paf = (mean_rr - 1) / mean_rr
        return paf

    pafs = pd.DataFrame([{'sex': sex,
                          'age_start': age_bin.left,
                          'age_end': age_bin.right,
                          'year_start': 2019,
                          'year_end': 2020,
                          'draw': input_draw,
                          'paf': calculate_paf_by_sex(sex)}
                         for sex in ['Female', 'Male']])
    return pafs


def write_pafs_to_hdf(config: str, artifact_path: str, output_dir: str, input_draw: str, random_seed: str):
    config = Path(config)
    output_dir = Path(output_dir)
    input_draw = int(input_draw)
    random_seed = int(random_seed)

    mkdir(output_dir, exists_ok=True)

    pafs = pd.concat([get_paf_for_age_group(config, artifact_path, input_draw, random_seed, age_group_id)
                      for age_group_id in metadata.AGE_GROUP.GBD_2019_LBWSG_RELATIVE_RISK])

    pafs.to_hdf(str(output_dir / f'draw_{input_draw}.hdf'), 'paf')


if __name__ == "__main__":
    write_pafs_to_hdf(*sys.argv[1:6])

from pathlib import Path
from typing import List, Tuple, Union

import click
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from vivarium.framework.randomness import get_hash
from vivarium_public_health.risks.data_transformations import pivot_categorical

from vivarium_gates_nutrition_optimization_child.constants import metadata

SeededDistribution = Tuple[str, stats.rv_continuous]


def len_longest_location() -> int:
    """Returns the length of the longest location in the project.

    Returns
    -------
       Length of the longest location in the project.
    """
    return len(max(metadata.LOCATIONS, key=len))


def sanitize_location(location: str):
    """Cleans up location formatting for writing and reading from file names.

    Parameters
    ----------
    location
        The unsanitized location name.

    Returns
    -------
        The sanitized location name (lower-case with white-space and
        special characters removed.

    """
    # FIXME: Should make this a reversible transformation.
    return location.replace(" ", "_").replace("'", "_").lower()


def delete_if_exists(*paths: Union[Path, List[Path]], confirm=False):
    paths = paths[0] if isinstance(paths[0], list) else paths
    existing_paths = [p for p in paths if p.exists()]
    if existing_paths:
        if confirm:
            # Assumes all paths have the same root dir
            root = existing_paths[0].parent
            names = [p.name for p in existing_paths]
            click.confirm(
                f"Existing files {names} found in directory {root}. Do you want to delete and replace?",
                abort=True,
            )
        for p in existing_paths:
            logger.info(f"Deleting artifact at {str(p)}.")
            p.unlink()


def get_norm(
    mean: float,
    sd: float = None,
    ninety_five_pct_confidence_interval: Tuple[float, float] = None,
) -> stats.norm:
    sd = _get_standard_deviation(mean, sd, ninety_five_pct_confidence_interval)
    return stats.norm(loc=mean, scale=sd)


def get_truncnorm(
    mean: float,
    sd: float = None,
    ninety_five_pct_confidence_interval: Tuple[float, float] = None,
    lower_clip: float = 0.0,
    upper_clip: float = 1.0,
) -> stats.norm:
    sd = _get_standard_deviation(mean, sd, ninety_five_pct_confidence_interval)
    a = (lower_clip - mean) / sd if sd else mean - 1e-03
    b = (upper_clip - mean) / sd if sd else mean + 1e03
    return stats.truncnorm(loc=mean, scale=sd, a=a, b=b)


def _get_standard_deviation(
    mean: float, sd: float, ninety_five_pct_confidence_interval: Tuple[float, float]
) -> float:
    if sd is None and ninety_five_pct_confidence_interval is None:
        raise ValueError(
            "Must provide either a standard deviation or a 95% confidence interval."
        )
    if sd is not None and ninety_five_pct_confidence_interval is not None:
        raise ValueError(
            "Cannot provide both a standard deviation and a 95% confidence interval."
        )
    if ninety_five_pct_confidence_interval is not None:
        lower = ninety_five_pct_confidence_interval[0]
        upper = ninety_five_pct_confidence_interval[1]
        if not (lower <= mean <= upper):
            raise ValueError(
                f"The mean ({mean}) must be between the lower ({lower}) and upper ({upper}) "
                "quantile values."
            )

        stdnorm_quantiles = stats.norm.ppf((0.025, 0.975))
        sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return sd


def get_truncnorm_from_sd(mean, sd, lower_clip: float = 0.0, upper_clip: float = 1.0):
    a = (lower_clip - mean) / sd
    b = (upper_clip - mean) / sd
    return stats.truncnorm(loc=mean, scale=sd, a=a, b=b)


def get_lognorm_from_quantiles(
    median: float, lower: float, upper: float, quantiles: Tuple[float, float] = (0.025, 0.975)
) -> stats.lognorm:
    """Returns a frozen lognormal distribution with the specified median, such that
    (lower, upper) are approximately equal to the quantiles with ranks
    (quantile_ranks[0], quantile_ranks[1]).
    """
    # Let Y ~ norm(mu, sigma^2) and X = exp(Y), where mu = log(median)
    # so X ~ lognorm(s=sigma, scale=exp(mu)) in scipy's notation.
    # We will determine sigma from the two specified quantiles lower and upper.
    if not (lower <= median <= upper):
        raise ValueError(
            f"The median ({median}) must be between the lower ({lower}) and upper ({upper}) "
            "quantile values."
        )

    # mean (and median) of the normal random variable Y = log(X)
    mu = np.log(median)
    # quantiles of the standard normal distribution corresponding to quantile_ranks
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    # quantiles of Y = log(X) corresponding to the quantiles (lower, upper) for X
    norm_quantiles = np.log([lower, upper])
    # standard deviation of Y = log(X) computed from the above quantiles for Y
    # and the corresponding standard normal quantiles
    sigma = (norm_quantiles[1] - norm_quantiles[0]) / (
        stdnorm_quantiles[1] - stdnorm_quantiles[0]
    )
    # Frozen lognormal distribution for X = exp(Y)
    # (s=sigma is the shape parameter; the scale parameter is exp(mu), which equals the median)
    return stats.lognorm(s=sigma, scale=median)


def get_random_variable_draws(columns: pd.Index, seed: str, distribution) -> pd.Series:
    return pd.Series(
        [get_random_variable(x, seed, distribution) for x in range(0, columns.size)],
        index=columns,
    )


def get_random_variable(draw: int, seed: str, distribution) -> pd.Series:
    np.random.seed(get_hash(f"{seed}_draw_{draw}"))
    return distribution.rvs()


def get_truncnorm_from_quantiles(
    mean: float,
    lower: float,
    upper: float,
    quantiles: Tuple[float, float] = (0.025, 0.975),
    lower_clip: float = 0.0,
    upper_clip: float = 1.0,
) -> stats.truncnorm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    a = (lower_clip - mean) / sd if sd else 0.0
    b = (upper_clip - mean) / sd if sd else 0.0
    return stats.truncnorm(loc=mean, scale=sd, a=a, b=b)


def get_norm_from_quantiles(
    mean: float, lower: float, upper: float, quantiles: Tuple[float, float] = (0.025, 0.975)
) -> stats.norm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return stats.norm(loc=mean, scale=sd)


def get_uniform_distribution_from_limits(lower_limit, upper_limit) -> stats.uniform:
    # stats.uniform is over [loc, loc+scal]
    loc = lower_limit
    scale = upper_limit - lower_limit
    return stats.uniform(loc=loc, scale=scale)

"""Shared utility functions for instance loading and parameter formatting."""

from src.distributions import NegExp, NorMal, GumBel, UniForm, BimodalNormal


def get_distribution_from_name(distr_name):
    """Instantiate a distribution object from its string name.

    Args:
        distr_name: One of 'NegExp', 'NorMal', 'GumBel', 'UniForm', 'BiNormal'.

    Returns:
        A distribution instance with default parameters.

    Raises:
        ValueError: If distr_name is not recognized.
    """
    distributions = {
        "NegExp": NegExp,
        "NorMal": NorMal,
        "GumBel": GumBel,
        "UniForm": UniForm,
        "BiNormal": BimodalNormal,
    }
    if distr_name not in distributions:
        raise ValueError(f"Invalid distribution name: {distr_name}")
    return distributions[distr_name]()


def format_cardinality(C):
    """Normalize a cardinality constraint to a (min, max) tuple.

    Args:
        C: Either an int (interpreted as (C, C)) or a tuple (min, max).

    Returns:
        A (min, max) tuple.

    Raises:
        ValueError: If C is neither an int nor a tuple.
    """
    if not isinstance(C, (int, tuple)):
        raise ValueError("C must be an integer or tuple")
    if isinstance(C, int):
        C = (C, C)
    return C

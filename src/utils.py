from src.distributions import NegExp, NorMal, GumBel, UniForm, BimodalNormal


def get_distribution_from_name(distr_name):
    if distr_name == "NegExp":
        return NegExp()
    elif distr_name == "NorMal":
        return NorMal()
    elif distr_name == "Gumbel":
        return GumBel()
    elif distr_name == "UniForm":
        return UniForm()
    elif distr_name == "BiNormal":
        return BimodalNormal()
    else:
        raise ValueError(f"Invalid distribution name: {distr_name}")
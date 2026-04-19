from src.distributions import NegExp, NorMal, GumBel, UniForm, BimodalNormal


def get_distribution_from_name(distr_name):
    if distr_name == "NegExp":
        return NegExp()
    elif distr_name == "NorMal":
        return NorMal()
    elif distr_name == "GumBel":
        return GumBel()
    elif distr_name == "UniForm":
        return UniForm()
    elif distr_name == "BiNormal":
        return BimodalNormal()
    else:
        raise ValueError(f"Invalid distribution name: {distr_name}")
    

def format_cardinality(C):
    if not isinstance(C, (int, tuple)):
        raise ValueError("C must be an integer or tuple")
    if isinstance(C, int):
        C = (C, C)
    return C
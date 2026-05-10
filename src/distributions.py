"""Random utility distributions for choice models.

Each distribution class implements cdf, pdf, and sample methods for a
zero-mean random variable used as the noise term in random utility models.
"""

import numpy as np
from scipy.stats import norm


class NegExp:
    """Negative exponential distribution (support on (-inf, 0]).

    Args:
        lmd: Rate parameter (default 1.0).
    """

    def __init__(self, lmd=1.0):
        self.lmd = lmd

    def get_name(self):
        return f"NegExp-{self.lmd}"

    def cdf(self, t):
        return np.minimum(np.exp(self.lmd * t), 1.0)

    def pdf(self, t):
        if t > 0.0:
            return 0.0
        else:
            return self.lmd * np.exp(self.lmd * t)

    def sample(self, size):
        return -np.random.exponential(1 / self.lmd, size)


class NorMal:
    """Normal distribution with zero mean.

    Args:
        std: Standard deviation (default 1.0).
    """

    def __init__(self, std=1.0):
        self.std = std

    def get_name(self):
        return f"NorMal-{self.std}"

    def cdf(self, t):
        return norm.cdf(t, loc=0.0, scale=self.std)

    def pdf(self, t):
        return norm.pdf(t, loc=0.0, scale=self.std)

    def sample(self, size):
        return np.random.normal(loc=0.0, scale=self.std, size=size)


class GumBel:
    """Gumbel distribution centered to zero mean via the Euler-Mascheroni constant.

    Args:
        eta: Scale parameter (default 1.0).
    """

    def __init__(self, eta=1.0):
        self.eta = eta

    def get_name(self):
        return f"GumBel-{self.eta}"

    def cdf(self, t):
        return np.exp(-np.exp(-(t / self.eta + np.euler_gamma)))

    def pdf(self, t):
        eta = self.eta
        z = t / eta + np.euler_gamma
        return (1 / eta) * np.exp(-z - np.exp(-z))

    def sample(self, size):
        eta = self.eta
        return -eta * np.euler_gamma - eta * np.log(-np.log(np.random.uniform(size=size)))


class UniForm:
    """Uniform distribution on [-delta, delta].

    Args:
        delta: Half-width of the support (default 1.0).
    """

    def __init__(self, delta=1.0):
        self.delta = delta

    def get_name(self):
        return f"UniForm-{self.delta}"

    def cdf(self, t):
        y = (t + self.delta) / (2 * self.delta)
        return np.clip(y, 0, 1)

    def pdf(self, t):
        if (-self.delta <= t) and (t <= self.delta):
            return 1 / (2 * self.delta)
        else:
            return 0.0

    def sample(self, size):
        return np.random.uniform(low=-self.delta, high=self.delta, size=size)


class BimodalNormal:
    """Bimodal mixture of two normals with zero overall mean.

    Args:
        loc: Distance of each mode from zero (default 3/sqrt(10)).
        p: Mixing weight for the left mode (default 0.5).
        std: Standard deviation of each component (default 1/sqrt(10)).
    """

    def __init__(self, loc=3.0 / np.sqrt(10), p=0.5, std=1.0 / np.sqrt(10)):
        self.loc = loc
        self.p = p
        self.std = std

    def get_name(self):
        return f"BimodalNormal-{self.loc}-{self.p}-{self.std}"

    def cdf(self, t):
        return (self.p * norm.cdf(t, loc=-self.loc, scale=self.std)
                + (1 - self.p) * norm.cdf(t, loc=self.loc, scale=self.std))

    def pdf(self, t):
        return (self.p * norm.pdf(t, loc=-self.loc, scale=self.std)
                + (1 - self.p) * norm.pdf(t, loc=self.loc, scale=self.std))

    def sample(self, size):
        normal_1 = np.random.normal(loc=-self.loc, scale=self.std, size=size)
        normal_2 = np.random.normal(loc=self.loc, scale=self.std, size=size)
        B = np.random.binomial(1, self.p, size=size)
        return B * normal_1 + (1 - B) * normal_2

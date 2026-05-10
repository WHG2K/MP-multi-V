"""Random instance generators for assortment optimization experiments.

Provides generators for single-class (IND, LINEAR) and mixture MNL instances
with configurable correlation structures and parameter ranges.
"""

import numpy as np


class InstanceGenerator:
    """Generate random problem instances with different correlation structures.

    Methods:
        IND: Independent utilities and revenues.
        LINEAR: Linearly correlated utilities and revenues.
        UNIFORM: Uniformly distributed utilities and revenues.
    """

    def IND(self, N, N0):
        """Generate an instance with independent utilities and revenues.

        Args:
            N: Number of products.
            N0: Number of outside options.

        Returns:
            Tuple of (u, r, v) as Python lists.
        """
        u = np.random.normal(loc=0.0, scale=1.0, size=N)
        r = np.random.uniform(low=10.0, high=100.0, size=N)
        v = np.random.normal(loc=0.0, scale=1.0, size=N0)
        return u.tolist(), r.tolist(), v.tolist()

    def LINEAR(self, N, N0, r_range=(10, 100), u_range=(-1.8, 1.8), std=0.5):
        """Generate an instance with linearly correlated utilities and revenues.

        Higher-revenue products tend to have lower utility (and vice versa),
        with Gaussian noise controlling the tightness of the relationship.

        Args:
            N: Number of products.
            N0: Number of outside options.
            r_range: Revenue range (min, max).
            u_range: Utility range (min, max).
            std: Standard deviation of the noise term.

        Returns:
            Tuple of (u, r, v) as Python lists.
        """
        q = (r_range[0] + r_range[1]) / 2
        p = (u_range[1] - u_range[0]) / (r_range[1] - r_range[0])

        r = np.random.uniform(low=r_range[0], high=r_range[1], size=N)
        eps = np.random.normal(loc=0.0, scale=std, size=N)
        u = -p * (r - q) + eps

        r_fake = np.random.uniform(low=r_range[0], high=r_range[1], size=N0)
        eps_fake = np.random.normal(loc=0.0, scale=std, size=N0)
        v = -p * (r_fake - q) + eps_fake

        return u.tolist(), r.tolist(), v.tolist()

    def UNIFORM(self, N, N0, r_range=(10, 100), u_range=(-1, 1)):
        """Generate an instance with uniformly distributed utilities and revenues.

        Args:
            N: Number of products.
            N0: Number of outside options.
            r_range: Revenue range (min, max).
            u_range: Utility range (min, max).

        Returns:
            Tuple of (u, r, v) as Python lists.
        """
        u = np.random.uniform(low=u_range[0], high=u_range[1], size=N)
        v = np.random.uniform(low=u_range[0], high=u_range[1], size=N0)
        r = np.random.uniform(low=r_range[0], high=r_range[1], size=N)
        return u.tolist(), r.tolist(), v.tolist()


class MixMNL_Generator:
    """Generate random instances for the mixture MNL model.

    Args:
        N: Number of products.
        K: Number of latent customer classes.
    """

    def __init__(self, N, K):
        self.N = N
        self.K = K

    def generate(self, r_range=(10, 100), u_bias=0.0, size_C=None, sort=False):
        """Generate one mixed-MNL instance.

        Args:
            r_range: Revenue range (min, max).
            u_bias: Additive bias for utility values.
            size_C: If set, sparsify preference factors so each class
                     considers only size_C products.
            sort: If True, sort utilities and revenues.

        Returns:
            Tuple of (pf, weights, r) where pf is a (K x N) preference
            factor matrix, weights is a length-K mixing probability vector,
            and r is a length-N revenue vector.
        """
        lmd = np.random.rand(self.K)
        weights = (lmd / lmd.sum()).tolist()

        u = np.random.randn(self.K, self.N)
        u += u_bias
        r = np.random.uniform(low=10.0, high=100.0, size=self.N)

        if sort:
            u = np.sort(u, axis=1)
            r = np.sort(r)[::-1].tolist()

        pf = np.exp(u)

        if size_C is not None:
            mask = np.zeros_like(pf, dtype=bool)
            for i in range(self.K):
                chosen = np.random.choice(self.N, size_C, replace=False)
                mask[i, chosen] = True
            pf = pf * mask

        pf = pf.tolist()
        return pf, weights, r

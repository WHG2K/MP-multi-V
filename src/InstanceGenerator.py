import numpy as np


class InstanceGenerator:

    def IND(self, N, N0):
        u = np.random.normal(loc=0.0, scale=1.0, size=N)
        r = np.random.uniform(low=10.0, high=100.0, size=N)
        v = np.random.normal(loc=0.0, scale=1.0, size=N0)

        return u.tolist(), r.tolist(), v.tolist()

    def LINEAR(self, N, N0, r_range=(10, 100), u_range=(-1.8, 1.8), std=0.5):
        # intersecpt
        q = (r_range[0]+r_range[1]) / 2
        p = (u_range[1]-u_range[0]) / (r_range[1]-r_range[0])
        # generate r and u
        r = np.random.uniform(low=r_range[0], high=r_range[1], size=N)
        eps = np.random.normal(loc=0.0, scale=std, size=N)
        u = -p*(r-q) + eps
        # want v to follow the same distribution as u
        r_fake = np.random.uniform(low=r_range[0], high=r_range[1], size=N0)
        eps_fake = np.random.normal(loc=0.0, scale=std, size=N0)
        v = -p*(r_fake-q) + eps_fake

        return u.tolist(), r.tolist(), v.tolist()
    

    def UNIFORM(self, N, N0, r_range=(10, 100), u_range=(-1, 1)):
        u = np.random.uniform(low=u_range[0], high=u_range[1], size=N)
        v = np.random.uniform(low=u_range[0], high=u_range[1], size=N0)
        r = np.random.uniform(low=r_range[0], high=r_range[1], size=N)

        return u.tolist(), r.tolist(), v.tolist()
    

class MixMNL_Generator:
    def __init__(self, N, K):
        self.N = N
        self.K = K

    def generate(self, r_range=(10, 100), u_bias=0.0, size_C=None, sort=False):
        """
        Generate one mixed-MNL instance:
        - u: a (K x N) array where each row u[k] is sampled ~ N(0,1)
        - pi: a length-K probability vector ~ Dirichlet(1,...,1)
        """
        # class mixing weights
        # class mixing weights as a list
        # p = np.random.dirichlet(alpha=np.ones(self.K)).tolist()
        lmd = np.random.rand(self.K)
        # 2) Normalize to sum to 1
        weights = (lmd / lmd.sum()).tolist()
        
        # utility weights for each latent class, as list of lists
        u = np.random.randn(self.K, self.N)
        u += u_bias
        r = np.random.uniform(low=10.0, high=100.0, size=self.N)
        if sort:
            u = np.sort(u, axis=1)
            r  = np.sort(r)[::-1].tolist()

        pf = np.exp(u)
        if size_C is not None:
            mask = np.zeros_like(pf, dtype=bool)
            for i in range(self.K):
                chosen = np.random.choice(self.N, size_C, replace=False)
                mask[i, chosen] = True
            pf = pf * mask
        pf = pf.tolist()
    
        
        return pf, weights, r

if __name__ == "__main__":

    params_generator = InstanceGenerator()

    N, B, C, N0 = (20, 4, 8, 3)  # K is n_pick, C is cardinality, N0 is N_outside

    #################################
    #### u and r are independent ####
    #################################
    u, r, v = params_generator.IND(N, N0)

    ########################################
    #### u and r are linearly dependent ####
    ########################################
    u, r, v = params_generator.LINEAR(N, N0)

    #########################################
    #### Generate a mixed-MNL instance ####
    #########################################
    pf, weights, r = MixMNL_Generator(N=6, K=3).generate(size_C=4)
    print(pf)
    print(weights)
    print(type(pf[0][1]))
    print(len(pf), len(pf[0]))
    print(r)
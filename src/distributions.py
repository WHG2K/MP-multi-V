import numpy as np
from scipy.stats import norm

class NegExp():
    def __init__(self, lmd=1.0):
        self.lmd = lmd

    def get_name(self):
        lmd = self.lmd
        return "NegExp-"+str(lmd)
    
    def cdf(self, t):
        return np.minimum(np.exp(self.lmd*t), 1.0)
    
    def pdf(self, t):
        if t>0.0:
            return 0.0
        else:
            return self.lmd*np.exp(self.lmd*t)
        
    def sample(self, size):
        lmd = self.lmd
        eps = -np.random.exponential(1/lmd, size)
        return eps

class NorMal(): # always assume mean 0
    def __init__(self, std=1.0):
        self.std = std

    def get_name(self):
        std = self.std
        return "NorMal-"+str(std)
    
    def cdf(self, t):
        return norm.cdf(t, loc=0.0, scale=self.std)
    
    def pdf(self, t):
        return norm.pdf(t, loc=0.0, scale=self.std)
    
    def sample(self, size):
        std = self.std
        eps = np.random.normal(loc=0.0, scale=std, size=size)
        return eps

class GumBel(): # using Euler-Mascheroni Constant to get zero mean
    def __init__(self, eta=1.0):
        self.eta = eta

    def get_name(self):
        eta = self.eta
        return "GumBel-"+str(eta)
    
    def cdf(self, t):
        return np.exp(-np.exp(-(t/self.eta+np.euler_gamma)))
    
    def pdf(self, t):
        eta = self.eta
        return 1/eta*np.exp(-(t/eta+np.euler_gamma)-np.exp(-(t/eta+np.euler_gamma)))
    
    def sample(self, size):
        eta = self.eta
        eps = -eta*np.euler_gamma - eta * np.log(-np.log(np.random.uniform(size=size)))
        return eps
        
class UniForm():
    def __init__(self, delta=1.0):
        self.delta = delta

    def get_name(self):
        delta = self.delta
        return "UniForm-"+str(delta)
    
    def cdf(self, t):
        delta = self.delta
        y = (t+delta)/(2*delta)
        y = np.clip(y, 0, 1)
        return y
    
    def pdf(self, t):
        delta = self.delta
        if (t>=-delta) and (t<=delta):
            return 1/(2*delta)
        else:
            return 0.0
        
    def sample(self, size):
        delta = self.delta
        eps = np.random.uniform(low=-delta, high=delta, size=size)
        return eps
    
class BimodalNormal():
    # def __init__(self, loc=2.0/np.sqrt(5), p=0.5, std=1.0/np.sqrt(5)):
    def __init__(self, loc=3.0/np.sqrt(10), p=0.5, std=1.0/np.sqrt(10)):
        self.loc = loc
        self.p = p
        self.std = std

    def get_name(self):
        loc = self.loc
        p = self.p
        std = self.std
        return "BimodalNormal-"+str(loc)+"-"+str(p)+"-"+str(std)
    
    def cdf(self, t):
        loc = self.loc
        p = self.p
        std = self.std
        return p*norm.cdf(t, loc=-loc, scale=std) + (1-p)*norm.cdf(t, loc=loc, scale=std)
    
    def pdf(self, t):
        loc = self.loc
        p = self.p
        std = self.std
        return p*norm.pdf(t, loc=-loc, scale=std) + (1-p)*norm.pdf(t, loc=loc, scale=std)

    def sample(self, size):
        loc = self.loc
        p = self.p
        std = self.std
        normal_1 = np.random.normal(loc=-loc, scale=std, size=size)
        normal_2 = np.random.normal(loc=loc, scale=std, size=size)
        B = np.random.binomial(1, p, size=size)
        eps = B*normal_1 + (1-B)*normal_2
        return eps
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import os
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm
import itertools
import matplotlib.pyplot as plt


class MNL():  # MNL for standard Gumbel with cardinality constraints
    def __init__(self, u, r, v, cardinality):
        '''
        All numpy arrays
        '''
        assert len(u)==len(r)
        self.u = np.array(u).reshape(-1)
        self.r = np.array(r).reshape(-1)
        self.v = v
        self.N = len(u)
        self.C = cardinality
        # normalize the utility of the no-purchase option to 0
        self.w = np.exp(u-v).reshape(-1)
        self.X = None
    
    def solve(self, verbose=0):
        N = self.N
        w = self.w
        r = self.r
        C = self.C

        # setup model
        m = gp.Model("mnl")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0

        # variables
        y = m.addMVar(N, lb=0, vtype=GRB.CONTINUOUS)
        y0 = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
        # objective
        m.setObjective((gp.quicksum(r[i]*y[i] for i in range(N))), GRB.MAXIMIZE)
        # basic constraint 1
        m.addConstr(y0 + gp.quicksum(y[i] for i in range(N)) <= 1)
        # cardinality constraint
        m.addConstr(gp.quicksum(y[i]/w[i] for i in range(N)) <= C*y0)
        # another set of constraints
        for j in range(N):
             m.addConstr(y[j] <= y0*w[j])
        # optimize
        m.optimize()

        if verbose==1:
            print("optimal objective value: ", m.objVal)

        self.X = np.where(y.X > 1e-8, 1, 0)
        return self.X

    def get_revenue(self, x):
        x = np.array(x).reshape(-1)
        w = self.w
        r = self.r
        N = self.N
        denominator = 1.0 + np.dot(w, x)
        purchase_prob = w*x/denominator
        return np.dot(r, purchase_prob)

    

class Exponomial():  # The exponomial model. No .solve() method (standard distribution wirth lmd=1)
    def __init__(self, u, r, v):
        self.u = np.array(u).reshape(-1)
        self.v = v
        self.r = np.array(r).reshape(-1)
        self.N = len(u)

    def get_revenue(self, x):
        N = self.N
        x = np.array(x).reshape(-1).tolist()
        # initialize v, r and y
        r_ = np.zeros(N+1)
        v_ = np.zeros(N+1)
        y_ = np.zeros(N+1)

        r_[0:N] = self.r    # The combined revenue vector
        v_[0:N] = self.u    # The combined utility vector
        v_[N] = self.v
        y_[0:N] = x
        y_[N] = 1

        # reorder by utility
        df = pd.DataFrame({'v': v_, 'r': r_, 'y': y_})
        df = df.sort_values(by=['v']).reset_index(drop=True)
        r_ = np.array(df['r'].values)
        v_ = np.array(df['v'].values)
        y_ = np.array(df['y'].values)

        # among choices
        r_ = r_[y_>0.9]
        v_ = v_[y_>0.9]

        # calculate probabilities of choosing, notations follows from Alptekinoglu & Semple (2016)
        m = len(v_)

        G = np.zeros(m)
        for i in range(m):
            G[i] = np.exp(-(v_[i:m].sum()-(m-i)*v_[i]))/(m-i)

        Q = np.zeros(m)  # probability of choosing product i
        for i in range(m):
            s = G[i]
            for j in range(i):
                s -= G[j]/(m-j-1)
            Q[i] = s

        return (Q*r_).sum()


class NegExp():
    def __init__(self, lmd=1.0):
        self.lmd = lmd
    def get_name(self):
        lmd = self.lmd
        return "NegExp-"+str(lmd)

class NorMal(): # always assume mean 0
    def __init__(self, std=1.0):
        self.std = std
    def get_name(self):
        std = self.std
        return "NorMal-"+str(std)

class GumBel(): # using Euler-Mascheroni Constant to get zero mean
    def __init__(self, eta=1.0):
        self.eta = eta
    def get_name(self):
        eta = self.eta
        return "GumBel-"+str(eta)
        
class UniForm():
    def __init__(self, delta=1.0):
        self.delta = delta
    def get_name(self):
        delta = self.delta
        return "UniForm-"+str(delta)
    
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



class GMCF():  # general multi-choice fluid model
    def __init__(self, u, r, v, n_pick, distribution, cardinality):
        assert len(u)==len(r)
        self.u = u
        self.r = r
        self.v = v
        self.distribution = distribution
        if not isinstance(cardinality, tuple):
            self.C = (0, cardinality)
        else:
            self.C = cardinality
        self.N = len(self.u)
        self.N_outside = len(self.v)
        self.alpha = (1.0/self.N*np.ones(self.N)).tolist()  # weights of each products. Default to be 1/N
        self.n_pick = n_pick

    def F_eps(self, t):
        if isinstance(t, list):
            t = np.array(t)
        if isinstance(self.distribution, NegExp):
            lmd = self.distribution.lmd
            return np.minimum(np.exp(lmd*t), 1.0)
        elif isinstance(self.distribution, NorMal):
            std = self.distribution.std
            return norm.cdf(t, loc=0.0, scale=std)
        elif isinstance(self.distribution, GumBel):
            eta = self.distribution.eta
            return np.exp(-np.exp(-(t/eta+np.euler_gamma)))
        elif isinstance(self.distribution, UniForm):
            delta = self.distribution.delta
            y = (t+delta)/(2*delta)
            y = np.clip(y, 0, 1)
            return y
        elif isinstance(self.distribution, BimodalNormal):
            loc = self.distribution.loc
            p = self.distribution.p
            std = self.distribution.std
            return p*norm.cdf(t, loc=-loc, scale=std) + (1-p)*norm.cdf(t, loc=loc, scale=std)
        else:
            print("Not an available distribution")

    def f_eps(self, t):
        if isinstance(self.distribution, NegExp):
            lmd = self.distribution.lmd
            if t>0.0:
                return 0.0
            else:
                return lmd*np.exp(lmd*t)
        elif isinstance(self.distribution, NorMal):
            std = self.distribution.std
            return norm.pdf(t, loc=0.0, scale=std)
        elif isinstance(self.distribution, GumBel):
            eta = self.distribution.eta
            return 1/eta*np.exp(-(t/eta+np.euler_gamma)-np.exp(-(t/eta+np.euler_gamma)))
        elif isinstance(self.distribution, UniForm):
            delta = self.distribution.delta
            if (t>=-delta) and (t<=delta):
                return 1/(2*delta)
            else:
                return 0.0
        elif isinstance(self.distribution, BimodalNormal):
            loc = self.distribution.loc
            p = self.distribution.p
            std = self.distribution.std
            return p*norm.pdf(t, loc=-loc, scale=std) + (1-p)*norm.pdf(t, loc=loc, scale=std)
        else:
            print("Not an available distribution")
    
    
    def IP_H(self, H, method=2, verbose=0):
        N = self.N
        N_outside = self.N_outside
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N_outside)

        if (method == 2):
            b = np.array([r[i]*(1-self.F_eps(H-u[i])) for i in range(N)])
            c = np.array([1-self.F_eps(H-u[i]) for i in range(N)])
            d = np.array([1-self.F_eps(H-v[j]) for j in range(N_outside)])


            # setup model
            m = gp.Model("P(H)")
            m.Params.OutputFlag = verbose
            m.Params.LogToConsole = verbose
            m.Params.Threads = 24

            x = m.addMVar(N, vtype=GRB.BINARY)
            # objective
            m.setObjective((gp.quicksum(b[i]*x[i] for i in range(N))), GRB.MAXIMIZE)
            # basic constraint
            m.addConstr(gp.quicksum(c[i]*x[i] for i in range(N)) + d.sum() <= self.n_pick)
            # cardinality constraint
            m.addConstr(gp.quicksum(x[i] for i in range(N))<=self.C[1])
            m.addConstr(gp.quicksum(x[i] for i in range(N))>=self.C[0])
            # optimize
            m.optimize()

            # print("number of solutions:", m.SolCount)

            if m.SolCount >= 1:
                return x.X, m.ObjVal/N
            else:
                return np.zeros(N), 0.0
            
        else:
            while True:
                b = np.array([r[i]*(1-self.F_eps(H-u[i])) for i in range(N)])
                c = np.array([1-self.F_eps(H-u[i]) for i in range(N)])
                d = np.array([1-self.F_eps(H-v[j]) for j in range(N_outside)])


                # setup model
                m = gp.Model("P(H)")
                m.Params.OutputFlag = 0
                m.Params.LogToConsole = 0

                x = m.addMVar(N, vtype=GRB.BINARY)
                # objective
                m.setObjective((gp.quicksum(b[i]*x[i] for i in range(N))), GRB.MAXIMIZE)
                # basic constraint
                m.addConstr(gp.quicksum(c[i]*x[i] for i in range(N)) + d.sum() <= self.n_pick)
                # cardinality constraint
                m.addConstr(gp.quicksum(x[i] for i in range(N))<=self.C[1])
                m.addConstr(gp.quicksum(x[i] for i in range(N))>=self.C[0])
                # optimize
                m.optimize()

                if m.SolCount >= 1:
                    return x.X, m.ObjVal/N
                else:
                    H += 1e-9

        
    
    def LP_H(self, H):
        N = self.N
        N_outside = self.N_outside
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N_outside)

        b = np.array([r[i]*(1-self.F_eps(H-u[i])) for i in range(N)])
        c = np.array([1-self.F_eps(H-u[i]) for i in range(N)])
        d = np.array([1-self.F_eps(H-v[j]) for j in range(N_outside)])


        # setup model
        m = gp.Model("P(H)")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0

        x = m.addMVar(N, lb=0, ub=1, vtype=GRB.CONTINUOUS)
        # objective
        m.setObjective((gp.quicksum(b[i]*x[i] for i in range(N))), GRB.MAXIMIZE)
        # basic constraint
        m.addConstr(gp.quicksum(c[i]*x[i] for i in range(N)) + d.sum() <= self.n_pick, name="Constr1")
        # cardinality constraint
        m.addConstr(gp.quicksum(x[i] for i in range(N))<=self.C[1])
        m.addConstr(gp.quicksum(x[i] for i in range(N))>=self.C[0])
        # optimize
        m.optimize()

        # get the gradient by envelope theorem
        if m.SolCount >= 1:
            primal = x.X
            dual = m.getConstrByName("Constr1").Pi
            gradient = sum([self.f_eps(H-u[i])*primal[i]*(dual-r[i]) for i in range(N)]) + dual*sum([self.f_eps(H-v[j]) for j in range(N_outside)])

            return m.ObjVal/N, gradient/N
        
        else:
            return -1, 10

    def find_h_range(self):  # used for i.i.d. epsilons
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        low1 = -10
        low2 = 10
        # u_min = min(self.u)
        min_size = max(1, self.C[0])
        u_sorted = np.sort(self.u)[0:min_size]
        for _ in range(30):
            low = (low1 + low2)/2
            if (sum([1-self.F_eps(low-u_sorted[i]) for i in range(min_size)]) + sum([1-self.F_eps(low-v[j]) for j in range(N_outside)]) >= n_pick):
                low1 = low
            else:
                low2 = low
        low = low2

        # print("error expected to be negative:", sum([1-self.F_eps(low-v[j]) for j in range(N_outside)])-n_pick)

        high1 = -10
        high2 = 10
        for _ in range(30):
            high = (high1 + high2)/2
            diff = sum([1-self.F_eps(high-u[i]) for i in range(N)]) + sum([1-self.F_eps(high-v[j]) for j in range(N_outside)]) - n_pick
            if (diff >= 0):
                high1 = high
            else:
                high2 = high
        high = high2
        return low, high



    def solve(self, method=2, n_steps=101):
        # find upper bound and lower bound
        # upper bound is achieved when offer all products
        # lower bound is achieved when offer 1 product that has the lowest utility (Assume U_i=u_i + (i.i.d.)eps_i)
        low, high = self.find_h_range()

        if (method==0):  # bisection search on LP
            for _ in range(30):
                mid = (low + high)/2
                _, grad = self.LP_H(mid)
                if grad>0:
                    low = mid
                else:
                    high = mid

            H = (low + high)/2
            x, _ = self.IP_H(H, method=0)
        elif (method==1):  # stepsize line search on LP
            # n_steps = 101
            h_arr = np.linspace(low, high, n_steps)
            best_h = -100
            best_rev = -1.0

            for h in h_arr:
                curr_rev, _ = self.LP_H(h)
                if (curr_rev > best_rev):
                    best_rev = curr_rev
                    best_h = h
            x, _ = self.IP_H(best_h, method=0)
        elif (method==2):  # stepsize line search on IP
            # n_steps = 101
            h_arr = np.linspace(low, high, n_steps)
            best_x = np.zeros(self.N)
            best_rev = -1.0
            # best_h = -100

            for h in h_arr:
                curr_x, curr_rev = self.IP_H(h, method=2)
                if (curr_rev > best_rev):
                    best_rev = curr_rev
                    best_x = curr_x
                    best_h = h # comment out
            x = best_x
            # print(f"w by line search is {best_h}")
        else:
            print("No such method.")
            return None

        x = np.array(x).reshape(-1)
        return np.round(x).astype(int)
    

    def solve_SP_UniForm(self):
        assert isinstance(self.distribution, UniForm)
        d = self.distribution.delta
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        # setup model
        m = gp.Model("IP formulation for UniForm")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0

        # range of w and p:=exp(lmd*w)
        wL, wH = self.find_h_range()

        # small enough delta_y for y, delta_z for z and M_w for w
        delta_y = np.array([1/(2*max(abs(wL-u[i]), abs(wH-u[i]))+2*d+10) for i in range(N)])
        delta_z = np.array([1/(2*max(abs(wL-v[j]), abs(wH-v[j]))+2*d+10) for j in range(N_outside)])

        # initialize variables
        w = m.addVar(lb=wL, ub=wH, vtype=GRB.CONTINUOUS)
        x = m.addMVar(N, vtype=GRB.BINARY)
        y = m.addMVar(N, vtype=GRB.BINARY)
        _y = m.addMVar(N, vtype=GRB.BINARY)
        y_ = m.addMVar(N, vtype=GRB.BINARY)
        z = m.addMVar(N_outside, vtype=GRB.BINARY)
        _z = m.addMVar(N_outside, vtype=GRB.BINARY)
        z_ = m.addMVar(N_outside, vtype=GRB.BINARY)
        q = m.addMVar(N, vtype=GRB.BINARY)
        _q = m.addMVar(N, vtype=GRB.BINARY)
        s = m.addMVar(N, lb=min(0,wL), ub=max(0,wH), vtype=GRB.CONTINUOUS)
        t = m.addMVar(N_outside, lb=min(0,wL), ub=max(0,wH), vtype=GRB.CONTINUOUS)

        # set objective
        # penalty = 1e-7
        # m.setObjective((gp.quicksum( r[i]*(d+u[i])/(2*d)*q[i] - r[i]/(2*d)*s[i] + r[i]*_q[i]  for i in range(N))) + gp.quicksum(penalty*x[i] for i in range(N)), GRB.MAXIMIZE)
        m.setObjective((gp.quicksum( r[i]*(d+u[i])/(2*d)*q[i] - r[i]/(2*d)*s[i] + r[i]*_q[i]  for i in range(N))), GRB.MAXIMIZE)

        # basic constraint
        m.addConstr(gp.quicksum( (d+u[i])/(2*d)*q[i] - 1/(2*d)*s[i] + _q[i]  for i in range(N)) 
                    + gp.quicksum( (d+v[j])/(2*d)*z[j] - 1/(2*d)*t[j] + _z[j]  for j in range(N_outside)) 
                    <= n_pick)

        # cardinality constraint
        m.addConstr(gp.quicksum(x[i] for i in range(N))<=self.C[1])
        m.addConstr(gp.quicksum(x[i] for i in range(N))>=self.C[0])

        # constraints for indicator variables _y, y and y_
        m.addConstrs((_y[i] <= 1 + delta_y[i]*(-d-w+u[i]) for i in range(N)))
        m.addConstrs((y_[i] <= 1 + delta_y[i]*(w-u[i]-d) for i in range(N)))
        m.addConstrs((y[i] <= 1 + delta_y[i]*(d-w+u[i]) for i in range(N)))
        m.addConstrs((y[i] <= 1 + delta_y[i]*(w-u[i]+d) for i in range(N)))
        m.addConstrs((_y[i] + y[i] + y_[i] == 1 for i in range(N)))

        # constraints for indicator variables _z, z and z_
        m.addConstrs((_z[j] <= 1 + delta_z[j]*(-d-w+v[j]) for j in range(N_outside)))
        m.addConstrs((z_[j] <= 1 + delta_z[j]*(w-v[j]-d) for j in range(N_outside)))
        m.addConstrs((z[j] <= 1 + delta_z[j]*(d-w+v[j]) for j in range(N_outside)))
        m.addConstrs((z[j] <= 1 + delta_z[j]*(w-v[j]+d) for j in range(N_outside)))
        m.addConstrs((_z[j] + z[j] + z_[j] == 1 for j in range(N_outside)))

        # constraints for product q[i]:=x[i]*y[i]
        m.addConstrs((q[i] <= x[i] for i in range(N)))
        m.addConstrs((q[i] <= y[i] for i in range(N)))
        m.addConstrs((q[i] >= x[i]+y[i]-1 for i in range(N)))

        # constraints for product q_[i]:=x[i]*y_[i]
        m.addConstrs((_q[i] <= x[i] for i in range(N)))
        m.addConstrs((_q[i] <= _y[i] for i in range(N)))
        m.addConstrs((_q[i] >= x[i]+_y[i]-1 for i in range(N)))

        # constraints for product s[i]=w*q[i] and t[j]=w*z[j]
        m.addConstrs((s[i] >= wL*q[i] for i in range(N)))
        m.addConstrs((s[i] <= wH*q[i] for i in range(N)))
        m.addConstrs((s[i] >= w-wH*(1-q[i]) for i in range(N)))
        m.addConstrs((s[i] <= w-wL*(1-q[i]) for i in range(N)))
        m.addConstrs((t[j] >= wL*z[j] for j in range(N_outside)))
        m.addConstrs((t[j] <= wH*z[j] for j in range(N_outside)))
        m.addConstrs((t[j] >= w-wH*(1-z[j]) for j in range(N_outside)))
        m.addConstrs((t[j] <= w-wL*(1-z[j]) for j in range(N_outside)))

        # optimize
        m.optimize()

        x = np.array(x.X).reshape(-1)
        return np.round(x).astype(int)
    
    
    def solve_SP_NegExp(self):
        assert isinstance(self.distribution, NegExp)
        lmd = self.distribution.lmd
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        # setup model
        m = gp.Model("IP formulation for NegExp")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0

        # range of w and p:=exp(lmd*w)
        wL, wH = self.find_h_range()
        pL = np.exp(lmd*wL)
        pH = np.exp(lmd*wH)

        # \bar{u} and \bar{v}
        bu = np.exp(lmd*u)
        bv = np.exp(lmd*v)

        # small enough delta_y for y and delta_z for z
        delta_y = np.array([1/(2*max(abs(pL-bu[i]), abs(pH-bu[i]))+10) for i in range(N)])
        delta_z = np.array([1/(2*max(abs(pL-bv[j]), abs(pH-bv[j]))+10) for j in range(N_outside)])

        # initialize variables
        p = m.addVar(lb=pL, ub=pH, vtype=GRB.CONTINUOUS)
        x = m.addMVar(N, lb=0, ub=1, vtype=GRB.BINARY)
        y = m.addMVar(N, lb=0, ub=1, vtype=GRB.BINARY)
        z = m.addMVar(N_outside, lb=0, ub=1, vtype=GRB.BINARY)
        q = m.addMVar(N, lb=0, ub=1, vtype=GRB.BINARY)
        s = m.addMVar(N, lb=min(0,pL), ub=max(0,pH), vtype=GRB.CONTINUOUS)
        t = m.addMVar(N_outside, lb=min(0,pL), ub=max(0,pH), vtype=GRB.CONTINUOUS)

        # set objective
        # penalty = 1e-5
        # m.setObjective((gp.quicksum(r[i]*(q[i]-1/bu[i]*s[i]) for i in range(N)) + gp.quicksum(penalty*x[i] for i in range(N))), GRB.MAXIMIZE)
        m.setObjective(gp.quicksum(r[i]*(q[i]-1/bu[i]*s[i]) for i in range(N)), GRB.MAXIMIZE)

        # basic constraint
        m.addConstr(gp.quicksum((q[i]-1/bu[i]*s[i]) for i in range(N)) 
                    + gp.quicksum((z[j]-1/bv[j]*t[j]) for j in range(N_outside)) 
                    <= n_pick)

        # cardinality constraint
        m.addConstr(gp.quicksum(x[i] for i in range(N))<=self.C[1])
        m.addConstr(gp.quicksum(x[i] for i in range(N))>=self.C[0])

        # constraints for indicator variables y and z
        m.addConstrs((y[i] <= 1+delta_y[i]*(bu[i]-p) for i in range(N)))
        m.addConstrs((y[i] >= delta_y[i]*(bu[i]-p) for i in range(N)))
        m.addConstrs((z[j] <= 1+delta_z[j]*(bv[j]-p) for j in range(N_outside)))
        m.addConstrs((z[j] >= delta_z[j]*(bv[j]-p) for j in range(N_outside)))

        # constraints for product h[i]:=x[i]*y[i]
        m.addConstrs((q[i] <= x[i] for i in range(N)))
        m.addConstrs((q[i] <= y[i] for i in range(N)))
        m.addConstrs((q[i] >= x[i]+y[i]-1 for i in range(N)))

        # constraints for product s[i]=p*q[i] and t[j]=p*z[j]
        m.addConstrs((s[i] >= pL*q[i] for i in range(N)))
        m.addConstrs((s[i] <= pH*q[i] for i in range(N)))
        m.addConstrs((s[i] >= p-pH*(1-q[i]) for i in range(N)))
        m.addConstrs((s[i] <= p-pL*(1-q[i]) for i in range(N)))
        m.addConstrs((t[j] >= pL*z[j] for j in range(N_outside)))
        m.addConstrs((t[j] <= pH*z[j] for j in range(N_outside)))
        m.addConstrs((t[j] >= p-pH*(1-z[j]) for j in range(N_outside)))
        m.addConstrs((t[j] <= p-pL*(1-z[j]) for j in range(N_outside)))

        # optimize
        m.optimize()

        x = np.array(x.X).reshape(-1)
        return np.round(x).astype(int)
    

    def LPIP_plot(self, n_steps=301, SP=True, RSP=True, save_path=None, format='png', show=True):
        # plt.style.use("ggplot")
        low, high = self.find_h_range()
        LP_arr = np.zeros(n_steps)
        IP_arr = np.zeros(n_steps)
        h_arr = np.linspace(low, high, n_steps)
        for i in range(len(h_arr)):
            h = h_arr[i]
            LP_arr[i], _ = self.LP_H(h)
            _, IP_arr[i] = self.IP_H(h)
        plt.figure()
        if RSP:
            plt.plot(h_arr, LP_arr, label="RSP(w)", color='red')
        if SP:
            plt.plot(h_arr, IP_arr, label="SP(w)", color='blue')
        plt.xlabel("w")
        plt.ylabel("value")
        plt.legend()
        if save_path is not None:
            if format=='pdf':
                plt.savefig(save_path, format='pdf')
            elif format=='png':
                plt.savefig(save_path, format='png', dpi=300)
        if show:
            plt.show()
        




'''
A simulator to simulate the revenue of a given problem instance.
'''


class GMCRevenueSimulator:
    def __init__(self, u, r, v, n_pick, distribution):
        assert len(u)==len(r)
        self.u = u
        self.r = r
        self.v = v
        self.N = len(u)
        self.N_outside = len(v)
        self.n_pick = n_pick
        self.distribution = distribution

    def Generate_batch(self, n_samples=10000):
        u = np.array(self.u).reshape(-1)
        v = np.array(self.v).reshape(-1)
        N = self.N
        N_outside = self.N_outside
        
        # generate a [n_samples, N+n_pick] sample matrix
        uv = np.hstack((u, v))
        uv1 = np.expand_dims(uv, axis=0)
        uv2 = np.repeat(uv1, n_samples, axis=0)

        if isinstance(self.distribution, NegExp):
            lmd = self.distribution.lmd
            eps = -np.random.exponential(1/lmd, [n_samples, N + N_outside])
        elif isinstance(self.distribution, NorMal):
            std = self.distribution.std
            eps = np.random.normal(loc=0.0, scale=std, size=[n_samples, N + N_outside])
        elif isinstance(self.distribution, GumBel):
            eta = self.distribution.eta
            eps = -eta*np.euler_gamma - eta * np.log(-np.log(np.random.uniform(size=[n_samples, N + N_outside])))
        elif isinstance(self.distribution, UniForm):
            delta = self.distribution.delta
            eps = np.random.uniform(low=-delta, high=delta, size=[n_samples, N + N_outside])
        elif isinstance(self.distribution, BimodalNormal):
            loc = self.distribution.loc
            p = self.distribution.p
            std = self.distribution.std
            normal_1 = np.random.normal(loc=-loc, scale=std, size=[n_samples, N + N_outside])
            normal_2 = np.random.normal(loc=loc, scale=std, size=[n_samples, N + N_outside])
            B = np.random.binomial(1, p, size=[n_samples, N + N_outside])
            eps = B*normal_1 + (1-B)*normal_2
        else:
            print("Not a valid distribution")
        
        return uv2+eps

    def Revenue(self, y_dict, n_samples=10000, given_data=None): # return data and confidence interval
        # allows for list, numpy array and dictionary
        if isinstance(y_dict, dict):
            is_dict = 1
        else:
            is_dict = 0
            y_dict = {"key1": y_dict}
            
        r = (self.r).copy()
        n_pick = self.n_pick
        N_outside = self.N_outside
        
        # create a new dictionary to store the batch revenues for each keys in y_dict
        r_dict = {}
        
        if given_data is None:
            data = self.Generate_batch(n_samples)
        else:
            data = given_data
            n_samples = given_data.shape[0]

        # calculate batch revenue
        for y_name in y_dict: 
            y = (y_dict[y_name]).copy()
            y_ = np.array(y).reshape(-1) # in case we don't know the input "y" is a list or numpy array
            
            y_ = np.hstack((y_, np.ones(self.N_outside)))
            data_y = data[:, y_>0.99]

            # Find the indices of the maximum element along the row axis
            v_max = np.zeros_like(data_y)
            sorted_indices = np.argsort(-data_y, axis=1) #descending order
            ranking = np.argsort(sorted_indices, axis=1)
            v_max[ranking <= n_pick-0.5] = 1

            # ger r[]
            r_ = np.array(r).reshape(-1)
            r_ = np.hstack((r_, np.zeros(N_outside)))
            r_ = r_[y_>0.99]
            r_mat = np.expand_dims(r_, axis=0)
            r_mat = np.repeat(r_mat, n_samples, axis=0)
            r_y = (r_mat*v_max).sum(axis=1)
            
            r_dict[y_name] = r_y
        
        # get mean and standard deviation
        df = pd.DataFrame(r_dict)
        df_mean = df.mean()
        df_std = df.std()
        
        mean_std = {}
            
        for y_name in df_mean.index:
            mean_std[y_name] = [df_mean[y_name], df_std[y_name]/np.sqrt(n_samples)]
            
        if is_dict:
            return r_dict, mean_std
        else:
            return r_dict["key1"], mean_std["key1"]
        


'''
ADXOPT heuristic
'''

class ADXOPT:
    def __init__(self, u, r, v, n_pick, distribution, cardinality):
        assert len(u)==len(r)
        self.u = u
        self.r = r
        self.v = v
        self.n_pick = n_pick
        self.distribution = distribution
        self.C = cardinality
    
    def solve(self, n_samples=10000, Revenue_func=None, verbose=False):
        N = len(self.u)
        C = self.C

        # initialize a revenue function
        if Revenue_func is None:
            revenue_simulator = GMCRevenueSimulator(self.u, self.r, self.v, self.n_pick, self.distribution)
            data = revenue_simulator.Generate_batch(n_samples=n_samples)
            def Revenue(S):
                y = np.zeros(N)
                y[S] = 1
                _, rslt = revenue_simulator.Revenue(y, given_data=data)
                return rslt[0]
        else:
            # The "n_samples" parameter won't be used in this case
            def Revenue(S):
                y = np.zeros(N)
                y[S] = 1
                return Revenue_func(y)

        # ADXOPT initialization
        S = []
        SA = []
        SA_ = []
        SD = []
        SD_ = []
        SX = []
        SX_ = []
        removal = np.zeros(N)
        b = 1

        while True:
            available_products = [j for j in range(N) if removal[j]<b and j not in S] # available products

            # calculate SA
            if len(S)==C:
                SA = []
            else:
                rev = -1
                for j in available_products:
                    SA_ = S.copy()
                    SA_.append(j)
                    if Revenue(SA_) > rev:
                        SA = SA_
                        rev = Revenue(SA_)

            if Revenue(SA) <= Revenue(S):
                # calculate SD
                rev = -1
                for i in S:
                    SD_ = S.copy()
                    SD_.remove(i)
                    if Revenue(SD_) > rev:
                        SD = SD_
                        rev = Revenue(SD_)
                # calculate SX
                rev = -1
                for i in S:
                    for j in available_products:
                        SX_ = S.copy()
                        SX_.append(j)
                        SX_.remove(i)
                        if Revenue(SX_) > rev:
                            SX = SX_
                            rev = Revenue(SX_)
                # determine S_next
                S_next = SD
                if Revenue(SX) > Revenue(SD):
                    S_next = SX
            else:
                S_next = SA

            # update removal counts
            for i in range(N):
                if i in S and i not in S_next:
                    removal[i] += 1

            if Revenue(S_next) <= Revenue(S) or min(removal)>=b:
                break

            S = S_next
        
            if verbose:
                S_print = S.copy()
                S_print.sort()
                print("Current: ", S_print, Revenue(S_print))

        # S_opt = S_next
        # if Revenue(S_next) < Revenue(S):
        #     S_opt = S
            
        y = np.zeros(N)
        y[S] = 1
        
        return y



'''
Revenue-ordered (RO) heuristic
'''

class RO:
    def __init__(self, u, r, v, n_pick, distribution, cardinality):
        assert len(u)==len(r)
        self.u = u
        self.r = r
        self.v = v
        self.n_pick = n_pick
        self.distribution = distribution
        self.C = cardinality
    
    def solve(self, n_samples=10000, Revenue_func=None):
        C = self.C
        N = len(self.u)

        # initialize a revenue function
        if Revenue_func is None:
            revenue_simulator = GMCRevenueSimulator(self.u, self.r, self.v, self.n_pick, self.distribution)
            data = revenue_simulator.Generate_batch(n_samples=n_samples)
            def Revenue(y):
                _, rslt = revenue_simulator.Revenue(y, given_data=data)
                return rslt[0]
        else:
            # The "n_samples" parameter won't be used in this case
            def Revenue(y):
                return Revenue_func(y)

        
        r_ = (self.r).copy()
        u_ = (self.u).copy()
        
        # sorted by revenue
        df = pd.DataFrame({'u': u_, 'r': r_})
        df = df.reset_index()
        df_ordered = df.sort_values(by=['r'], ascending=False)      # (inplace=True)
        
        current_opt = 0.0
        y_best = np.zeros(N)
        
        # for non-zeros
        for num in range(C):
            # nested-in-revenue solutions
            y = np.zeros(N)
            y[0:num+1] = 1
            
            # recover
            df_ordered_ = df_ordered.copy()
            df_ordered_['y'] = y
            df2 = df_ordered_.sort_values(by=['index'], ascending=True)
            y = df2['y'].values
            
            # compare revenues
            revenue = Revenue(y)
            
            if revenue>current_opt:
                current_opt = revenue
                y_best = y.copy()
                
        return np.round(y_best)



'''
Backward Elimination heuristic
-> Introduced in Aydın and John 2016 for the Exponomial model.
-> But the algorithm can be applied to any distribution.
-> Must be full assrtment, i.e. cardinality=N
'''

class BackwardElim:
    def __init__(self, u, v, r, n_pick, distribution):
        self.u = u
        self.v = v
        self.r = r
        self.n_pick = n_pick
        self.distribution = distribution
    
    def solve(self, n_samples=10000, Revenue_func=None):
        N = len(self.u)

        # initialize a revenue function
        if Revenue_func is None:
            revenue_simulator = GMCRevenueSimulator(self.u, self.r, self.v, self.n_pick, self.distribution)
            data = revenue_simulator.Generate_batch(n_samples=n_samples)
            def Revenue(S):
                y = np.zeros(N)
                y[S] = 1
                _, rslt = revenue_simulator.Revenue(y, given_data=data)
                return rslt[0]
        else:
            # Revenue = Revenue_func
            def Revenue(S):
                y = np.zeros(N)
                y[S] = 1
                return Revenue_func(y)

        S = [i for i in range(N)]

        while len(S) > 0:
            
            rev = Revenue(S)
            S_next = None
            for j in S:
                S_ = S.copy()
                S_.remove(j)
                if Revenue(S_) > rev:
                    S_next = S_
                    rev = Revenue(S_)

            if S_next is not None:
                S = S_next
            else:
                break
            
        y = np.zeros(N)
        y[S] = 1
        
        return y
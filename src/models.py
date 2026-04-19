import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm
from src.distributions import UniForm, NegExp
import pandas as pd
from src.utils import format_cardinality



class MPMVSurrogate():
    def __init__(self, u, r, v, n_pick, distr, C, solver_params=None):
        assert len(u)==len(r)
        self.u = u
        self.r = r
        self.v = v
        self.distr = distr
        if not isinstance(C, tuple):
            self.C = (0, C)
        else:
            self.C = C
        self.N = len(self.u)
        self.N_outside = len(self.v)
        self.n_pick = n_pick
        self.solver_params = solver_params or {
            'Threads': 24,
            # 'MIPGap': 1e-6,
            'MIPGapAbs': 0,
            # 'IntFeasTol': 1e-6,
            # 'OptimalityTol': 1e-6,
            # 'FeasibilityTol': 1e-6,
            'Heuristics': 0,
        }


    def _configure_model(self, m):
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        for key, val in self.solver_params.items():
            m.setParam(key, val)
    
    
    def SP(self, w, verbose=0):
        N = self.N
        N_outside = self.N_outside
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N_outside)

        b = np.array([r[i]*(1-self.distr.cdf(w-u[i])) for i in range(N)])
        c = np.array([1-self.distr.cdf(w-u[i]) for i in range(N)])
        d = np.array([1-self.distr.cdf(w-v[j]) for j in range(N_outside)])


        # setup model
        m = gp.Model("SP(w)")
        # m.Params.OutputFlag = 0
        # m.Params.LogToConsole = 0
        # m.Params.Threads = 24
        self._configure_model(m)

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
            return [0 for _ in range(N)], 0.0
        
    
    def RSP(self, w):
        N = self.N
        N_outside = self.N_outside
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N_outside)

        b = np.array([r[i]*(1-self.distr.cdf(w-u[i])) for i in range(N)])
        c = np.array([1-self.distr.cdf(w-u[i]) for i in range(N)])
        d = np.array([1-self.distr.cdf(w-v[j]) for j in range(N_outside)])


        # setup model
        m = gp.Model("P(H)")
        # m.Params.OutputFlag = 0
        # m.Params.LogToConsole = 0
        # m.Params.Threads = 24
        self._configure_model(m)

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
            # dual = m.getConstrByName("Constr1").Pi
            # gradient = sum([self.distr.pdf(H-u[i])*primal[i]*(dual-r[i]) for i in range(N)]) + dual*sum([self.distr.pdf(H-v[j]) for j in range(N_outside)])

            # return m.ObjVal/N, gradient/N
            return x.X, m.ObjVal/N
        
        else:
            # return -1, 10
            return [0.0 for _ in range(N)], 0.0


    def _get_box_range(self):
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        # r = np.array(self.r)
        v = np.array(self.v)

        low1 = -100
        low2 = 100
        # u_min = min(self.u)
        min_size = max(1, self.C[0])
        u_sorted = np.sort(self.u)[0:min_size]
        for _ in range(30):
            low = (low1 + low2)/2
            if (sum([1-self.distr.cdf(low-u_sorted[i]) for i in range(min_size)]) + sum([1-self.distr.cdf(low-v[j]) for j in range(N_outside)]) >= n_pick):
                low1 = low
            else:
                low2 = low
        low = low2

        # print("error expected to be negative:", sum([1-self.distr.cdf(low-v[j]) for j in range(N_outside)])-n_pick)

        high1 = -100
        high2 = 100
        for _ in range(30):
            high = (high1 + high2)/2
            diff = sum([1-self.distr.cdf(high-u[i]) for i in range(N)]) + sum([1-self.distr.cdf(high-v[j]) for j in range(N_outside)]) - n_pick
            if (diff >= 0):
                high1 = high
            else:
                high2 = high
        high = high2
        return low, high



    def solve(self, method='SP', n_steps=1001):
        # find upper bound and lower bound
        low, high = self._get_box_range()

        if (method=='RSP'):  # stepsize line search on RSP(w)
            h_arr = np.linspace(low, high, n_steps)
            best_h = -100
            best_rev = -1.0

            for h in h_arr:
                _, curr_rev = self.RSP(h)
                if (curr_rev > best_rev):
                    best_rev = curr_rev
                    best_h = h
            x, _ = self.SP(best_h)
        elif (method=='SP'):  # stepsize line search on SP(w)

            if isinstance(self.distr, UniForm):
                return self._solve_SP_UniForm_MILP()
            elif isinstance(self.distr, NegExp):
                return self._solve_SP_NegExp_MILP()
            # n_steps = 101
            h_arr = np.linspace(low, high, n_steps)
            best_x = np.zeros(self.N)
            best_rev = -1.0

            for h in h_arr:
                curr_x, curr_rev = self.SP(h)
                if (curr_rev > best_rev):
                    best_rev = curr_rev
                    best_x = curr_x
                    # best_h = h # comment out
            x = best_x

        else:
            print("No such method.")
            return None

        x = np.array(x).reshape(-1)
        return np.round(x).astype(int).tolist()
    

    def _solve_SP_UniForm_MILP(self):
        assert isinstance(self.distr, UniForm)
        d = self.distr.delta
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        # setup model
        m = gp.Model("IP formulation for UniForm")
        # m.Params.OutputFlag = 0
        # m.Params.LogToConsole = 0
        # m.Params.Threads = 24
        self._configure_model(m)

        # range of w and p:=exp(lmd*w)
        wL, wH = self._get_box_range()

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
        return np.round(x).astype(int).tolist()
    
    
    def _solve_SP_NegExp_MILP(self):
        assert isinstance(self.distr, NegExp)
        lmd = self.distr.lmd
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        r = np.array(self.r)
        v = np.array(self.v)

        # setup model
        m = gp.Model("IP formulation for NegExp")
        # m.Params.OutputFlag = 0
        # m.Params.LogToConsole = 0
        # m.Params.Threads = 24
        self._configure_model(m)

        # range of w and p:=exp(lmd*w)
        wL, wH = self._get_box_range()
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
        return np.round(x).astype(int).tolist()


    def _w_x_(self, x):
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        v = np.array(self.v)

        w_low = -10
        w_high = 10

        for _ in range(30):
            mid = (w_low + w_high)/2
            if (sum([(1-self.distr.cdf(mid-u[i]))*x[i] for i in range(N)]) + sum([(1-self.distr.cdf(mid-v[j])) for j in range(N_outside)]) >= n_pick):
                w_low = mid
            else:
                w_high = mid
        
        return w_high

    def _pi_hat_(self, x):
        w_x = self._w_x_(x)
        r = np.array(self.r)
        u = np.array(self.u)
        N = self.N

        b = np.array([r[i]*(1-self.distr.cdf(w_x-u[i])) for i in range(N)])

        return sum([b[i]*x[i] for i in range(N)])
    
    def __call__(self, x):
        return self._pi_hat_(x)



class MPMVOriginal:
    def __init__(self, u, r, v, n_pick, distr):
        assert len(u)==len(r)
        self.u = u
        self.r = r
        self.v = v
        self.N = len(u)
        self.N_outside = len(v)
        self.n_pick = n_pick
        self.distr = distr
        self.random_comp = None

    def Generate_batch(self, n_samples=10000):
        u = np.array(self.u).reshape(-1)
        v = np.array(self.v).reshape(-1)
        N = self.N
        N_outside = self.N_outside
        
        # generate a [n_samples, N+n_pick] sample matrix
        uv = np.hstack((u, v))
        uv1 = np.expand_dims(uv, axis=0)
        uv2 = np.repeat(uv1, n_samples, axis=0)

        eps = self.distr.sample(size=[n_samples, N + N_outside])
        
        return uv2+eps
    
    def set_random_comp(self, random_comp):
        self.random_comp = random_comp

    def Revenue(self, y): # return data and confidence interval
            
        r = (self.r).copy()
        n_pick = self.n_pick
        N_outside = self.N_outside

        if self.random_comp is None:
            self.set_random_comp(self.Generate_batch(n_samples=10000))
        else:
            data = self.random_comp
            n_samples = data.shape[0]
        
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

        return float(r_y.mean())
    
    def __call__(self, x):
        return self.Revenue(x)
    





class MixedSP:    # mixed u and v, but deterministic B (n_pick)

    def __init__(self, u, r, v, n_pick, distr, C, weights):
        assert len(u[0])==len(r)
        assert len(u)==len(weights)
        self.u_all = u
        self.r = r
        self.v_all = v
        self.distr = distr
        if not isinstance(C, tuple):
            self.C = (0, C)
        else:
            self.C = C
        self.N = len(self.u_all[1])
        self.N_outside = len(self.v_all[1])
        self.n_pick = n_pick
        self.weights = weights
        self.K = len(weights)

    def _get_box_range(self):
        N = self.N
        K = self.K
        N_outside = self.N_outside
        n_pick = self.n_pick
        u_all = np.array(self.u_all)
        v_all = np.array(self.v_all)
        low_list = []
        high_list = []

        for k in range(K):
            u = u_all[k]
            v = v_all[k]

            low1 = -10
            low2 = 10
            min_size = max(1, self.C[0])
            u_sorted = np.sort(u)[0:min_size]
            for _ in range(30):
                low = (low1 + low2)/2
                if (sum([1-self.distr.cdf(low-u_sorted[i]) for i in range(min_size)]) + sum([1-self.distr.cdf(low-v[j]) for j in range(N_outside)]) >= n_pick):
                    low1 = low
                else:
                    low2 = low
            low = low2
            low_list.append(low)

            # print("error expected to be negative:", sum([1-self.distr.cdf(low-v[j]) for j in range(N_outside)])-n_pick)

            high1 = -10
            high2 = 10
            for _ in range(30):
                high = (high1 + high2)/2
                diff = sum([1-self.distr.cdf(high-u[i]) for i in range(N)]) + sum([1-self.distr.cdf(high-v[j]) for j in range(N_outside)]) - n_pick
                if (diff >= 0):
                    high1 = high
                else:
                    high2 = high
            high = high2
            high_list.append(high)

        min_low = min(low_list)
        max_high = max(high_list)
        return min_low, max_high
        # return [[mix_low, mix_high] for _ in range(K)]

    def SP(self, w, verbose=0):
        N = self.N
        N_outside = self.N_outside
        K = self.K
        weights = self.weights
        assert len(w) == K
        u_all = np.array(self.u_all)
        r = np.array(self.r)
        v_all = np.array(self.v_all)
        w_vec = w   # change name to w_vec. Later use w for scalar.

        # setup model
        m = gp.Model("SP(w)")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        m.Params.Threads = 24

        b_list = []
        c_list = []
        d_list = []

        for k in range(K):
            u = u_all[k]
            v = v_all[k]
            w = w_vec[k]

            b_list.append(np.array([r[i]*(1-self.distr.cdf(w-u[i])) for i in range(N)]))
            c_list.append(np.array([1-self.distr.cdf(w-u[i]) for i in range(N)]))
            d_list.append(np.array([1-self.distr.cdf(w-v[j]) for j in range(N_outside)]))
        
        b_mat = np.array(b_list)
        weights_arr = np.array(weights)

        b = (weights_arr @ b_mat).tolist()

        x = m.addMVar(N, vtype=GRB.BINARY)
        # objective
        m.setObjective((gp.quicksum(b[i]*x[i] for i in range(N))), GRB.MAXIMIZE)
        # basic constraint
        for k in range(K):
            c = c_list[k]
            d = d_list[k]
            m.addConstr(gp.quicksum(c[i]*x[i] for i in range(N)) + d.sum() <= self.n_pick)
        # cardinality constraint
        m.addConstr(gp.quicksum(x[i] for i in range(N))<=self.C[1])
        m.addConstr(gp.quicksum(x[i] for i in range(N))>=self.C[0])
        # optimize
        m.optimize()

        # print("number of solutions:", m.SolCount)

        if m.SolCount >= 1:
            return [int(round(xi)) for xi in x.X], m.ObjVal
        else:
            return [0 for _ in range(N)], 0.0









class MixtureSP:  # we might solve it using herustics.
    def __init__(self, models, weights):
        assert len(models) == len(weights)
        assert abs(sum(weights) - 1.0) <= 1e-6
        self.models = models
        self.weights = weights

    # def SP(self, w):
    #     assert len(w) == len(self.models)
    #     rslt = 0
    #     for i in range(len(self.models)):
    #         model = self.models[i]
    #         w = self.weights[i]
    #         rslt += w*model.SP(w)
    #     return rslt

    def __call__(self, x):
        rslt = 0
        for i in range(len(self.models)):
            model = self.models[i]
            w = self.weights[i]
            rslt += w*model(x)
        return rslt
    




# class MNL():  # MNL for standard Gumbel with cardinality constraints
#     def __init__(self, u, r, v, C):
#         '''
#         All numpy arrays
#         '''
#         assert len(u)==len(r)
#         self.u = np.array(u).reshape(-1)
#         self.r = np.array(r).reshape(-1)
#         self.v = v
#         self.N = len(u)
#         self.C = C
#         # normalize the utility of the no-purchase option to 0
#         self.w = np.exp(u-v).reshape(-1)
#         self.X = None
    
#     def solve(self, verbose=0):
#         N = self.N
#         w = self.w
#         r = self.r
#         C = self.C

#         # setup model
#         m = gp.Model("mnl")
#         m.Params.OutputFlag = 0
#         m.Params.LogToConsole = 0

#         # variables
#         y = m.addMVar(N, lb=0, vtype=GRB.CONTINUOUS)
#         y0 = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
#         # objective
#         m.setObjective((gp.quicksum(r[i]*y[i] for i in range(N))), GRB.MAXIMIZE)
#         # basic constraint 1
#         m.addConstr(y0 + gp.quicksum(y[i] for i in range(N)) <= 1)
#         # cardinality constraint
#         m.addConstr(gp.quicksum(y[i]/w[i] for i in range(N)) <= C*y0)
#         # another set of constraints
#         for j in range(N):
#              m.addConstr(y[j] <= y0*w[j])
#         # optimize
#         m.optimize()

#         if verbose==1:
#             print("optimal objective value: ", m.objVal)

#         self.X = np.where(y.X > 1e-8, 1, 0)
#         return self.X

#     def get_revenue(self, x):
#         x = np.array(x).reshape(-1)
#         w = self.w
#         r = self.r
#         N = self.N
#         denominator = 1.0 + np.dot(w, x)
#         purchase_prob = w*x/denominator
#         return np.dot(r, purchase_prob)
    
#     def __call__(self, x):
#         return self.get_revenue(x)







class MixtureMNL: # not sure if K=1 could cause problem.

    def __init__(self, pf, r, weights):
        # self.u = np.array(u).tolist()
        self.pf = pf # need formatting
        self.r = np.array(r).reshape(-1).tolist()
        self.weights = np.array(weights).reshape(-1).tolist()
        self.N = len(r)
        self.K = len(pf)
        # self.pf = np.exp(np.array(self.u)).tolist()

        # print("pf=", self.pf)

        assert len(weights) == self.K, f"weights must have the same row length as u, got {len(weights)} and {self.K}"
        assert abs(sum(weights) - 1.0) <= 1e-6, f"weights must sum to 1, got {sum(weights)}"
        assert len(pf[0]) == self.N, f"u must have the same column length as r, got {len(pf[0])} and {self.N}"


    def get_revenue(self, x):
        x = np.array(x).reshape(-1).tolist()
        pf = self.pf
        weights = self.weights
        r = self.r
        N = self.N
        K = self.K

        # the following code can be vectorized, but not need to do it rn.
        total_rev = 0.0
        for k in range(K):
            denom = 1.0 + sum(pf[k][i] * x[i] for i in range(N))
            numerator = sum(r[i] * pf[k][i] * x[i] for i in range(N))
            total_rev += weights[k] * numerator / denom

        return float(total_rev)
        

    def solve(self, C=None, verbose=0):

        pf = self.pf
        weights = self.weights
        r = self.r
        N = self.N
        K = self.K
        if C is not None:
            C = format_cardinality(C)

        m = gp.Model("mixture_mnl")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        m.Params.Threads = 24

        x = m.addMVar(N, vtype=GRB.BINARY)
        y = m.addMVar(K, lb= 0, vtype=GRB.CONTINUOUS)
        z = m.addMVar(shape=(K,N), lb=0, vtype=GRB.CONTINUOUS)

        # objective
        obj = gp.quicksum(
            weights[k] * r[i] * pf[k][i] * z[k, i]
            for k in range(K) for i in range(N)
        )
        m.setObjective(obj, GRB.MAXIMIZE)

        # constraints
        for k in range(K):
            m.addConstr(y[k] + gp.quicksum(pf[k][j] * z[k, j] for j in range(N)) == 1)
            for i in range(N):
                m.addConstr(y[k] - z[k, i] <= (1 - x[i]))
                m.addConstr(z[k, i] <= y[k])
                m.addConstr(z[k, i] <= x[i])

        if C is not None:
            m.addConstr(gp.quicksum(x[i] for i in range(N)) <= C[1])
            m.addConstr(gp.quicksum(x[i] for i in range(N)) >= C[0])

        # optimize
        m.optimize()
        x_opt = x.X
        return [int(round(xi)) for xi in x_opt], m.objVal
    

    def solve_space_constr(self, s, W=None, verbose=0):

        pf = self.pf
        weights = self.weights
        r = self.r
        N = self.N
        K = self.K
        s = np.array(s).reshape(-1)

        m = gp.Model("mixture_mnl")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        m.Params.Threads = 24

        x = m.addMVar(N, vtype=GRB.BINARY)
        y = m.addMVar(K, lb= 0, vtype=GRB.CONTINUOUS)
        z = m.addMVar(shape=(K,N), lb=0, vtype=GRB.CONTINUOUS)

        # objective
        obj = gp.quicksum(
            weights[k] * r[i] * pf[k][i] * z[k, i]
            for k in range(K) for i in range(N)
        )
        m.setObjective(obj, GRB.MAXIMIZE)

        # constraints
        for k in range(K):
            m.addConstr(y[k] + gp.quicksum(pf[k][j] * z[k, j] for j in range(N)) == 1)
            for i in range(N):
                m.addConstr(y[k] - z[k, i] <= (1 - x[i]))
                m.addConstr(z[k, i] <= y[k])
                m.addConstr(z[k, i] <= x[i])

        if W is not None:
            m.addConstr(gp.quicksum(s[i]*x[i] for i in range(N)) <= W)

        # optimize
        m.optimize()
        x_opt = x.X
        return [int(round(xi)) for xi in x_opt], m.objVal



    
    def __call__(self, x):
        return self.get_revenue(x)





class MNL_Space_Constr():  # MNL for standard Gumbel with cardinality constraints
    # def __init__(self, u, r, v, weights, W):
    def __init__(self, pf, r):
        # assert len(u)==len(r)
        self.pf = np.array(pf).reshape(-1)
        self.r = np.array(r).reshape(-1)
        # self.v0 = v0
        self.N = len(pf)
        # self.c = c
        # self.W = W
        # normalize the utility of the no-purchase option to 0
        # self.w = np.exp(u-v).reshape(-1)
        self.X = None

    def solve(self, s, W, verbose=0):
        # space constraint sum s_i*x_i <= W
        N = self.N
        pf = self.pf
        r = self.r
        # weights = self.weights
        s = np.array(s).reshape(-1)

        # setup model
        m = gp.Model("mnl_space_constr")
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        m.Params.Threads = 24
        m.Params.MIPGap = 1e-6

        # variables
        x = m.addMVar(N, vtype=GRB.BINARY)
        y = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS)   # y = 1/(1+\sum w_ix_i)
        # objective
        m.setObjective((gp.quicksum(r[i]*pf[i]*x[i]*y for i in range(N))), GRB.MAXIMIZE)
        # constraint for fraction y
        m.addConstr(y + gp.quicksum(pf[i]*x[i]*y for i in range(N)) <= 1)
        # space constraint
        m.addConstr(gp.quicksum(s[i]*x[i] for i in range(N)) <= W)

        # optimize
        m.optimize()

        if verbose==1:
            print("optimal objective value: ", m.objVal)

        self.X = np.array(x.X, dtype=int).tolist()
        return self.X

    def get_revenue(self, x):
        x = np.array(x).reshape(-1)
        pf = self.pf
        r = self.r
        # N = self.N
        denominator = 1.0 + np.dot(pf, x)
        purchase_prob = pf * x / denominator
        return np.dot(r, purchase_prob)
    
    def __call__(self, x):
        return self.get_revenue(x)



class MPMVSurrogate_Space_Constr():  # general multi-choice fluid model
    def __init__(self, u, r, v, n_pick, distr, s, W, solver_params=None):
        self.u = u
        self.r = r
        self.v = v
        self.s = s  # weights or size of each products
        self.W = W  # space limit
        self.distr = distr
        self.N = len(self.u)
        self.N_outside = len(self.v)
        self.n_pick = n_pick
        self.solver_params = solver_params or {
            'Threads': 24,
            # 'MIPGap': 1e-6,
            'MIPGapAbs': 0,
            # 'IntFeasTol': 1e-6,
            # 'OptimalityTol': 1e-6,
            # 'FeasibilityTol': 1e-6,
            'Heuristics': 0,
        }


    def _configure_model(self, m):
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        for key, val in self.solver_params.items():
            m.setParam(key, val)

    def SP_space(self, w):
        N = self.N
        N_outside = self.N_outside
        u = np.array(self.u).reshape(-1)
        r = np.array(self.r).reshape(-1)
        v = np.array(self.v).reshape(-1)
        s = np.array(self.s).reshape(-1)

        b = np.array([r[i]*(1-self.distr.cdf(w-u[i])) for i in range(N)])
        c = np.array([1-self.distr.cdf(w-u[i]) for i in range(N)])
        d = np.array([1-self.distr.cdf(w-v[j]) for j in range(N_outside)])

        # setup model
        m = gp.Model("SP(w)")
        # m.Params.OutputFlag = 0
        # m.Params.LogToConsole = 0
        # m.Params.Threads = 24
        self._configure_model(m)

        x = m.addMVar(N, vtype=GRB.BINARY)
        # objective
        m.setObjective((gp.quicksum(b[i]*x[i] for i in range(N))), GRB.MAXIMIZE)
        # basic constraint
        m.addConstr(gp.quicksum(c[i]*x[i] for i in range(N)) + d.sum() <= self.n_pick)
        # space constraint
        m.addConstr(gp.quicksum(s[i]*x[i] for i in range(N))<=self.W)
        # optimize
        m.optimize()

        # print("number of solutions:", m.SolCount)

        if m.SolCount >= 1:
            return x.X, m.ObjVal/N
        else:
            return [0 for _ in range(N)], 0.0

    def _get_box_range_space_constr(self):
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        # r = np.array(self.r)
        v = np.array(self.v)

        low1 = -10
        low2 = 10
        u_min = min(self.u)
        # min_size = max(1, self.C[0])
        # u_sorted = np.sort(self.u)[0:min_size]
        for _ in range(30):
            low = (low1 + low2)/2
            if (1-self.distr.cdf(low-u_min) + sum([1-self.distr.cdf(low-v[j]) for j in range(N_outside)]) >= n_pick):
                low1 = low
            else:
                low2 = low
        low = low2

        # print("error expected to be negative:", sum([1-self.distr.cdf(low-v[j]) for j in range(N_outside)])-n_pick)

        high1 = -10
        high2 = 10
        for _ in range(30):
            high = (high1 + high2)/2
            diff = sum([1-self.distr.cdf(high-u[i]) for i in range(N)]) + sum([1-self.distr.cdf(high-v[j]) for j in range(N_outside)]) - n_pick
            if (diff >= 0):
                high1 = high
            else:
                high2 = high
        high = high2
        return low, high
    
    def solve_space_constr(self, n_steps=1001):
        # find upper bound and lower bound
        low, high = self._get_box_range_space_constr()

        # if isinstance(self.distr, UniForm):
        #     return self._solve_SP_UniForm_MILP()
        # elif isinstance(self.distr, NegExp):
        #     return self._solve_SP_NegExp_MILP()
        
        h_arr = np.linspace(low, high, n_steps)
        best_x = np.zeros(self.N)
        best_rev = -1.0

        for h in h_arr:
            curr_x, curr_rev = self.SP_space(h)
            if (curr_rev > best_rev):
                best_rev = curr_rev
                best_x = curr_x
                # best_h = h # comment out
        x = best_x

        x = np.array(x).reshape(-1)
        return np.round(x).astype(int).tolist()
    

    def _w_x_(self, x):
        N = self.N
        N_outside = self.N_outside
        n_pick = self.n_pick
        u = np.array(self.u)
        v = np.array(self.v)

        w_low = -10
        w_high = 10

        for _ in range(30):
            mid = (w_low + w_high)/2
            if (sum([(1-self.distr.cdf(mid-u[i]))*x[i] for i in range(N)]) + sum([(1-self.distr.cdf(mid-v[j])) for j in range(N_outside)]) >= n_pick):
                w_low = mid
            else:
                w_high = mid
        
        return w_high

    def _pi_hat_(self, x):
        w_x = self._w_x_(x)
        r = np.array(self.r)
        u = np.array(self.u)
        N = self.N

        b = np.array([r[i]*(1-self.distr.cdf(w_x-u[i])) for i in range(N)])

        return sum([b[i]*x[i] for i in range(N)])
    
    def __call__(self, x):
        return self._pi_hat_(x)
    



class MixedSP_Space_Constr:    # mixed u and v, but deterministic B (n_pick)

    def __init__(self, u, r, v, n_pick, distr, weights, s, W, solver_params=None):
        assert len(u[0])==len(r)
        assert len(u)==len(weights)
        self.u_all = u
        self.r = r
        self.v_all = v
        self.distr = distr
        self.s = s
        self.W = W
        self.N = len(self.u_all[1])
        self.N_outside = len(self.v_all[1])
        self.n_pick = n_pick
        self.weights = weights
        self.K = len(weights)
        self.solver_params = solver_params or {
            'Threads': 24,
            # 'MIPGap': 1e-6,
            'MIPGapAbs': 0,
            # 'IntFeasTol': 1e-6,
            # 'OptimalityTol': 1e-6,
            # 'FeasibilityTol': 1e-6,
            'Heuristics': 0,
        }


    def _configure_model(self, m):
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        for key, val in self.solver_params.items():
            m.setParam(key, val)

    def _get_box_range_space_constr(self):
        N = self.N
        K = self.K
        N_outside = self.N_outside
        n_pick = self.n_pick
        u_all = np.array(self.u_all)
        v_all = np.array(self.v_all)
        low_list = []
        high_list = []

        for k in range(K):
            u = u_all[k]
            v = v_all[k]

            low1 = -10
            low2 = 10
            min_size = 1
            u_sorted = np.sort(u)[0:min_size]
            for _ in range(30):
                low = (low1 + low2)/2
                if (sum([1-self.distr.cdf(low-u_sorted[i]) for i in range(min_size)]) + sum([1-self.distr.cdf(low-v[j]) for j in range(N_outside)]) >= n_pick):
                    low1 = low
                else:
                    low2 = low
            low = low2
            low_list.append(low)

            # print("error expected to be negative:", sum([1-self.distr.cdf(low-v[j]) for j in range(N_outside)])-n_pick)

            high1 = -10
            high2 = 10
            for _ in range(30):
                high = (high1 + high2)/2
                diff = sum([1-self.distr.cdf(high-u[i]) for i in range(N)]) + sum([1-self.distr.cdf(high-v[j]) for j in range(N_outside)]) - n_pick
                if (diff >= 0):
                    high1 = high
                else:
                    high2 = high
            high = high2
            high_list.append(high)

        min_low = min(low_list)
        max_high = max(high_list)
        return min_low, max_high
        # return [[mix_low, mix_high] for _ in range(K)]

    def SP_space_constr(self, w, verbose=0):
        N = self.N
        N_outside = self.N_outside
        K = self.K
        weights = self.weights
        s = self.s
        assert len(w) == K
        u_all = np.array(self.u_all)
        r = np.array(self.r)
        v_all = np.array(self.v_all)
        w_vec = w   # change name to w_vec. Later use w for scalar.

        # setup model
        m = gp.Model("SP(w)")
        # m.Params.OutputFlag = 0
        # m.Params.LogToConsole = 0
        # m.Params.Threads = 24
        self._configure_model(m)

        b_list = []
        c_list = []
        d_list = []

        for k in range(K):
            u = u_all[k]
            v = v_all[k]
            w = w_vec[k]

            b_list.append(np.array([r[i]*(1-self.distr.cdf(w-u[i])) for i in range(N)]))
            c_list.append(np.array([1-self.distr.cdf(w-u[i]) for i in range(N)]))
            d_list.append(np.array([1-self.distr.cdf(w-v[j]) for j in range(N_outside)]))
        
        b_mat = np.array(b_list)
        weights_arr = np.array(weights)

        b = (weights_arr @ b_mat).tolist()

        x = m.addMVar(N, vtype=GRB.BINARY)
        # objective
        m.setObjective((gp.quicksum(b[i]*x[i] for i in range(N))), GRB.MAXIMIZE)
        # basic constraint
        for k in range(K):
            c = c_list[k]
            d = d_list[k]
            m.addConstr(gp.quicksum(c[i]*x[i] for i in range(N)) + d.sum() <= self.n_pick)
        # cardinality constraint
        m.addConstr(gp.quicksum(s[i]*x[i] for i in range(N))<=self.W)
        # optimize
        m.optimize()

        # print("number of solutions:", m.SolCount)

        if m.SolCount >= 1:
            return [int(round(xi)) for xi in x.X], m.ObjVal
        else:
            return [0 for _ in range(N)], 0.0

    






class MNL():  # MNL for standard Gumbel with cardinality constraints
    def __init__(self, pf, r):
        assert len(pf)==len(r)
        self.pf = np.array(pf).reshape(-1)
        self.r = np.array(r).reshape(-1)
        self.N = len(pf)
        # self.C = C
        # normalize the utility of the no-purchase option to 0
        # self.w = np.exp(u-v).reshape(-1)
        self.X = None
    
    def solve(self, C, verbose=0):
        N = self.N
        pf = self.pf
        r = self.r
        # C = self.C

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
        m.addConstr(gp.quicksum(y[i]/pf[i] for i in range(N)) <= C*y0)
        # another set of constraints
        for j in range(N):
             m.addConstr(y[j] <= y0*pf[j])
        # optimize
        m.optimize()

        if verbose==1:
            print("optimal objective value: ", m.objVal)

        self.X = np.where(y.X > 1e-8, 1, 0)
        return self.X

    def get_revenue(self, x):
        x = np.array(x).reshape(-1)
        pf = self.pf
        r = self.r
        N = self.N
        denominator = 1.0 + np.dot(pf, x)
        purchase_prob = pf * x / denominator
        return np.dot(r, purchase_prob)
    
    def __call__(self, x):
        return self.get_revenue(x)


import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm
from src.distributions import UniForm, NegExp


class MPMVSurrogate():
    def __init__(self, u, r, v, n_pick, distr, C):
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
        self.alpha = (1.0/self.N*np.ones(self.N)).tolist()  # weights of each products. Default to be 1/N
        self.n_pick = n_pick
    
    
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
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
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
            return [0 for _ in range(N)], 0.0
            
        # while True:
        #     b = np.array([r[i]*(1-self.distr.cdf(w-u[i])) for i in range(N)])
        #     c = np.array([1-self.distr.cdf(w-u[i]) for i in range(N)])
        #     d = np.array([1-self.distr.cdf(w-v[j]) for j in range(N_outside)])


        #     # setup model
        #     m = gp.Model("SP(w)")
        #     m.Params.OutputFlag = 0
        #     m.Params.LogToConsole = 0

        #     x = m.addMVar(N, vtype=GRB.BINARY)
        #     # objective
        #     m.setObjective((gp.quicksum(b[i]*x[i] for i in range(N))), GRB.MAXIMIZE)
        #     # basic constraint
        #     m.addConstr(gp.quicksum(c[i]*x[i] for i in range(N)) + d.sum() <= self.n_pick)
        #     # cardinality constraint
        #     m.addConstr(gp.quicksum(x[i] for i in range(N))<=self.C[1])
        #     m.addConstr(gp.quicksum(x[i] for i in range(N))>=self.C[0])
        #     # optimize
        #     m.optimize()

        #     if m.SolCount >= 1:
        #         return x.X, m.ObjVal/N
        #     else:
        #         H += 1e-9

        
    
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
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0
        m.Params.Threads = 24

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

        low1 = -10
        low2 = 10
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



    def solve(self, method='SP', n_steps=1001):
        # find upper bound and lower bound
        low, high = self._get_box_range()

        if (method=='RSP'):  # stepsize line search on RSP(w)
            h_arr = np.linspace(low, high, n_steps)
            best_h = -100
            best_rev = -1.0

            for h in h_arr:
                curr_rev, _ = self.RSP(h)
                if (curr_rev > best_rev):
                    best_rev = curr_rev
                    best_h = h
            x, _ = self.SP(best_h, method=0)
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
    

    # def LPIP_plot(self, n_steps=301, SP=True, RSP=True, save_path=None, format='png', show=True):
    #     # plt.style.use("ggplot")
    #     low, high = self._get_box_range()
    #     LP_arr = np.zeros(n_steps)
    #     IP_arr = np.zeros(n_steps)
    #     h_arr = np.linspace(low, high, n_steps)
    #     for i in range(len(h_arr)):
    #         h = h_arr[i]
    #         LP_arr[i], _ = self.RSP(h)
    #         _, IP_arr[i] = self.SP(h)
    #     plt.figure()
    #     if RSP:
    #         plt.plot(h_arr, LP_arr, label="RSP(w)", color='red')
    #     if SP:
    #         plt.plot(h_arr, IP_arr, label="SP(w)", color='blue')
    #     plt.xlabel("w")
    #     plt.ylabel("value")
    #     plt.legend()
    #     if save_path is not None:
    #         if format=='pdf':
    #             plt.savefig(save_path, format='pdf')
    #         elif format=='png':
    #             plt.savefig(save_path, format='png', dpi=300)
    #     if show:
    #         plt.show()
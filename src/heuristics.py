import numpy as np
from typing import Tuple, Union, Callable, List
from abc import ABC, abstractmethod
import pandas as pd
from src.utils import format_cardinality


class Heuristic(ABC):
    def __init__(self, N: int):
        """Initialize the optimizer
        
        Args:
            N: Number of products
            C: Cardinality constraint, can be an integer or (min, max) tuple
            num_cores: Number of CPU cores to use for parallel computation
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.N = N
        

    @abstractmethod
    def maximize(self, objective_fn: Callable, **kwargs) -> Tuple[List[int], float]:
        pass


'''
ADXOPT heuristic
'''

class ADXOPT(Heuristic):

    def maximize(self, objective_fn: Callable, **kwargs) -> Tuple[List[int], float]:

        verbose = kwargs.get("verbose", 0)
        assert verbose in [0, 1]

        N = self.N
        C = kwargs.get("C", None)
        if C is not None:
            C = format_cardinality(C)
        else:
            C = (0, N)
        assert C[0] == 0, f"We can only handle C[0]={0} for now."
        C = C[1] 

        def Revenue(S): # as a function of sets
            y = np.zeros(N)
            y[S] = 1
            return objective_fn(y)

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

        # format output
        x = np.array(y).reshape(-1).astype(int).tolist()
        fx = float(objective_fn(x))
        return x, fx
    



'''
Revenue-ordered (RO) heuristic
'''

class RO(Heuristic):
    
    def maximize(self, objective_fn: Callable, **kwargs) -> Tuple[List[int], float]:
        """
        Revenue-ordered heuristic:
         1. Sort items by revenue r in descending order.
         2. For k = 1..C, incrementally add items and evaluate the objective.
         3. Return the best solution x (as list of ints) and its objective value (float).

        Args:
            objective_fn: function mapping a 0–1 numpy array (length N) to a float
            kwargs must contain:
              - r: array-like of length N, revenues for ordering

        Returns:
            x_best_list (List[int]): best 0–1 solution vector of length N
            v_best_float (float): the corresponding objective value
        """
        N = self.N
        C = kwargs.get("C", None)
        if C is not None:
            C = format_cardinality(C)
        else:
            C = (0, N)
        assert C[0] == 0, f"We can only handle C[0]={0} for now."
        C = C[1]

        r = kwargs.get("r", None)
        assert r is not None, "You must pass revenue vector `r`"
        r = np.asarray(r)
        assert r.shape == (N,), f"`r` must be length N={N}"

        # sort indices by revenue descending
        sorted_idx = np.argsort(r)[::-1]

        # build x incrementally as numpy array
        x = np.zeros(N, dtype=int)
        v_best = -np.inf
        x_best = x.copy()

        # flip one bit at a time and evaluate
        for idx in sorted_idx[: C]:
            x[idx] = 1
            v = objective_fn(x)
            if v > v_best:
                v_best = v
                x_best = x.copy()

        # convert outputs to required types
        x = np.array(x_best).reshape(-1).astype(int).tolist()
        fx = float(objective_fn(x))
        return x, fx
    


class Greedy(Heuristic):
    """Greedy algorithm for solving optimization problems with cardinality constraints.
    
    Attributes:
        N: Number of products
        C: Cardinality constraint (min, max)
        num_cores: Number of CPU cores to use for parallel computation
    """

    def _one_step_search_list(self, x):
        if x.sum() == self.N:
            raise ValueError("No more solutions to search")
        else:
            # generate all possible solutions by adding to x
            x = np.round(x).astype(int)
            next_searches = []
            for i in range(len(x)):
                if x[i] == 0:  # 仅考虑当前为 0 的位置
                    y = x.copy()
                    y[i] = 1
                    next_searches.append(y)
            return next_searches


    def maximize(self, objective_fn: Callable, **kwargs) -> Tuple[List[int], float]:
        """Maximize the objective function using a greedy algorithm
        
        Args:
            objective_fn: Objective function to maximize
            max_iter: Maximum number of iterations
        """

        N = self.N
        C = kwargs.get("C", None)
        if C is not None:
            C = format_cardinality(C)
        else:
            C = (0, N)

        # star from zero solution
        x = np.zeros(N, dtype=int)
        if C[0] > 0:
            global_val = float('-inf')
        else:
            global_val = objective_fn(x)


        while x.sum() < C[1]:
            next_searches = self._one_step_search_list(x)
            best_val = float('-inf')

            for y in next_searches:
                val = objective_fn(y)
                if val > best_val:
                    best_val = val
                    best_y = y
            
            # print(f"layer: {best_y.sum()}, best_val: {best_val}, global_val: {global_val}")
            
            if (best_y.sum() > C[0]) and (best_val < global_val):
                return x, objective_fn(x)
            else:
                global_val = best_val
                x = best_y

        x = np.array(x).reshape(-1).astype(int).tolist()
        fx = float(objective_fn(x))
        return x, fx
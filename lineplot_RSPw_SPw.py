import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import MPMVSurrogate
from src.InstanceGenerator import InstanceGenerator
from src.utils import get_distribution_from_name


def LPIP_plot(surrogate, n_steps=301, SP=True, RSP=True,
              save_path=None, format='png', show=True, title=None):
    """
    Plot RSP(w) (LP relaxation) and SP(w) (IP) value curves over w
    for a given MPMVSurrogate instance.
    """
    low, high = surrogate._get_box_range()
    h_arr = np.linspace(low, high, n_steps)

    LP_arr = np.zeros(n_steps)
    IP_arr = np.zeros(n_steps)

    for i, h in enumerate(h_arr):
        if RSP:
            _, LP_arr[i] = surrogate.RSP(h)
        if SP:
            _, IP_arr[i] = surrogate.SP(h)

    plt.figure()
    if RSP:
        plt.plot(h_arr, LP_arr, label="RSP(w)", color='red')
    if SP:
        plt.plot(h_arr, IP_arr, label="SP(w)", color='blue')
    plt.xlabel("w")
    plt.ylabel("value")
    if title is not None:
        plt.title(title)
    plt.legend()

    if save_path is not None:
        if format == 'pdf':
            plt.savefig(save_path, format='pdf')
        elif format == 'png':
            plt.savefig(save_path, format='png', dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':

    path = "./data/RSPw2SPw/"
    os.makedirs(path, exist_ok=True)

    N_list = [20, 30, 40, 50]

    # Generate a large pool of parameters up front, slice per N
    np.random.seed(0)
    instance_generator = InstanceGenerator()
    u_all, r_all, v_all = instance_generator.IND(N=100, N0=100)
    F = "GumBel"

    solver_params = {
        'Threads': 24,
        # 'MIPGap':         1e-6,
        # 'MIPGapAbs':      0,
        # 'IntFeasTol':     1e-6,
        # 'OptimalityTol':  1e-6,
        # 'FeasibilityTol': 1e-6,
        # 'Heuristics':     0.05,
    }

    for N in tqdm(N_list):
        B           = round(0.1 * N)
        N0          = round(0.1 * N)
        C           = round(0.2 * N)
        cardinality = (0, C)

        # Instance
        u     = u_all[0:N]
        r     = r_all[0:N]
        v     = v_all[0:N0]
        distr = get_distribution_from_name(F)

        # Build surrogate and plot SP/RSP curves
        sp = MPMVSurrogate(u, r, v, B, distr, cardinality, solver_params=solver_params)
        save_file = os.path.join(
            path, f"RSP(w)_SP(w)_l_N_{N}.pdf"
        )
        LPIP_plot(sp, save_path=save_file, format='pdf', show=False)
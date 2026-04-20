import json
from tqdm import tqdm
from src.models import MixtureMNL, MPMVSurrogate, MixtureSP, MixedSP_Space_Constr
from src.distributions import GumBel
from src.heuristics import ADXOPT, RO, Greedy
import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import time
from skopt import forest_minimize, gp_minimize, plots
from skopt.space import Real
import matplotlib.pyplot as plt
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with customizable parameters.")
    parser.add_argument('--node', type=str, default=None, help='which node to run on')
    parser.add_argument("--data", type=str, default="", help='data file name')
    return parser.parse_args()


class OBJ_FUNC_SP_Space_Constr:

    def __init__(self, sp_model):
        self.sp_model = sp_model

    def __call__(self, w):
        return -self.sp_model.SP_space_constr(w)[1]


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # Choose environment
    if not args.node:
        load_dotenv(override=True)
    else:
        load_dotenv(override=True, dotenv_path=f'{args.node}.env')

    filename = args.data

    with open(filename, 'r', encoding='utf-8') as f:
        total_instances = sum(1 for _ in f)

    updated_instances = []

    with open(filename, 'r', encoding='utf-8') as f:

        for line in tqdm(f, total=total_instances, desc="Redo SP"):

            inst = json.loads(line)
            N = inst['N']
            K = inst['K']
            pf = inst['pf']
            r = inst['r']
            s = inst['s']
            W = inst['W']
            weights = inst['weights']
            C = tuple(inst['C'])

            mixmnl = MixtureMNL(pf, r, weights)
            u_mix = np.log(pf).tolist()



            time_4 = time.time()
            x_milp, val_milp = mixmnl.solve_space_constr(s=s, W=W)
            time_5 = time.time()

            time_6 = time.time()
            mixedSP_space_constr = MixedSP_Space_Constr(u_mix, r, [[0.0] for _ in range(K)], 1, GumBel(), weights, s, W)

            low, high = mixedSP_space_constr._get_box_range_space_constr()
            space = [Real(low, high) for _ in range(K)]
            obj_func_sp_space_constr = OBJ_FUNC_SP_Space_Constr(mixedSP_space_constr)

            res = gp_minimize(
                func=obj_func_sp_space_constr,
                dimensions=space,
                n_calls=200,            # Total number of evaluations
                n_initial_points=20,    # Initial random samples
                acq_func="EI",          # Expected Improvement
                random_state=0
            )

            # === Optional debug output ===
            # print("Best objective value:", res.fun)
            # print("Best parameters:")
            # for name, val in zip([d.name for d in space], res.x):
            #     print(f"  {name} = {val:.6f}")

            # === Optional visualization ===
            # # 7.1 Acquisition function curve
            # plots.plot_gaussian_process(res)
            # plt.title("GP surrogate & acquisition (EI)")
            # plt.show()
            # # 7.2 Pairwise parameter-objective plot
            # plots.plot_objective(res)
            # plt.show()

            w_baye = res.x
            x_sp_baye = mixedSP_space_constr.SP_space_constr(w_baye)[0]
            x_sp_baye = [int(round(x)) for x in x_sp_baye]

            time_7 = time.time()

            inst['x_milp'] = x_milp
            inst['pi_milp'] = mixmnl(x_milp)
            inst['time_milp'] = time_5 - time_4

            inst["x_baye"]    = x_sp_baye
            inst["pi_baye"]   = mixmnl(x_sp_baye)
            inst["time_baye"] = time_7 - time_6

            updated_instances.append(inst)

    root, ext = os.path.splitext(filename)
    output_path = f"{root}_spaceconstr_solved{ext}"

    updated_df = pd.DataFrame(updated_instances)
    updated_df.to_json(
        output_path,
        orient='records',
        lines=True,
        force_ascii=False,
    )
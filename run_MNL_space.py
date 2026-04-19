import json
from tqdm import tqdm
from src.models_new import MNL_Space_Constr, MPMVSurrogate_Space_Constr
from src.distributions import GumBel
from src.heuristics import ADXOPT, RO, Greedy
import argparse
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with customizable parameters.")
    
    parser.add_argument('--node', type=str, default=None, help='which node to run on')
    parser.add_argument("--data", type=str, default="", help='data file name')

    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # choose environment
    if not args.node:
        load_dotenv(override=True)
    else:
        load_dotenv(override=True, dotenv_path=f'{args.node}.env')
        
    filename = args.data

    with open(filename, 'r', encoding='utf-8') as f:
        total_instances = sum(1 for _ in f)

    updated_instances = []

    solver_params = {
            'Threads': 24,
            # 'MIPGap': 1e-6,
            # 'MIPGapAbs': 0,
            # 'IntFeasTol': 1e-6,
            # 'OptimalityTol': 1e-6,
            # 'FeasibilityTol': 1e-6,
            # 'Heuristics': 0.05,
        }

    with open(filename, 'r', encoding='utf-8') as f:

        for line in tqdm(f, total=total_instances, desc="Redo SP"):
            
            inst = json.loads(line)
            N = inst['N']
            pf = inst['pf']
            r = inst['r']
            s = inst['s']
            C = inst['C']

            mnl_space = MNL_Space_Constr(pf, r)
            # time_1 = time.time()
            # ro = RO(N)
            # x_ro, val_ro = ro.maximize(mnl_space, C=(0, C), r=r)
            time_2 = time.time()
            x_milp = mnl_space.solve(s=s, W=C)
            time_3 = time.time()
            u = np.log(pf).tolist()
            sp = MPMVSurrogate_Space_Constr(u, r, [0.0], 1, GumBel(), s, C, solver_params=solver_params)
            x_sp = sp.solve_space_constr(n_steps=1001)
            time_4 = time.time()

            # inst["x_ro"] = x_ro
            inst["x_milp"] = x_milp
            inst["x_sp"] = x_sp

            # inst["pi_ro"] = mnl_space(x_ro)
            inst["pi_milp"] = mnl_space(x_milp)
            inst["pi_sp"] = mnl_space(x_sp)

            # inst["time_ro"] = time_2 - time_1
            inst["time_milp"] = time_3 - time_2
            inst["time_sp"] = time_4 - time_3

            updated_instances.append(inst)

            # break


    root, ext = os.path.splitext(filename)
    output_path = f"{root}_spaceconstr_solved{ext}"

    updated_df = pd.DataFrame(updated_instances)
    updated_df.to_json(
        output_path,
        orient='records',
        lines=True,
        force_ascii=False,
    )

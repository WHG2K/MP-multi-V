# Solve SP and brute-force optimal for each instance. Produces Figures 2, L2.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from tqdm import tqdm
from src.models import MPMVSurrogate, MPMVOriginal
from src.utils import get_distribution_from_name
from src.brute_force import BruteForceOptimizer
import argparse
from dotenv import load_dotenv
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
            'Threads': os.cpu_count(),
        }

    with open(filename, 'r', encoding='utf-8') as f:

        for line in tqdm(f, total=total_instances, desc="SP"):
            
            inst = json.loads(line)
            N = inst['N']
            u = inst['u']
            r = inst['r']
            v = inst['v']
            n_pick = inst['B']
            distr = get_distribution_from_name(inst['F'])
            C = tuple(inst['C'])

            sp = MPMVSurrogate(u, r, v, n_pick, distr, C, solver_params=solver_params)
            time_1 = time.time()
            x_sp = sp.solve(method='SP', n_steps=3001)
            time_2 = time.time()
            op = MPMVOriginal(u, r, v, n_pick, distr)
            random_comp = op.Generate_batch(n_samples=10000)
            op.set_random_comp(random_comp)
            bf_solver = BruteForceOptimizer(N, C, num_cores=os.cpu_count())
            x_bf, pi_bf = bf_solver.maximize(op)


            random_comp = op.Generate_batch(n_samples=100000)
            op.set_random_comp(random_comp)

            pi_bf = op.Revenue(x_bf)
            pi_sp = op.Revenue(x_sp)

            inst["x_sp"] = np.array(x_sp).tolist()
            inst["time_sp"] = time_2 - time_1

            inst["pi_bf"] = pi_bf
            inst["pi_sp"] = pi_sp

            updated_instances.append(inst)


    root, ext = os.path.splitext(filename)
    output_path = f"{root}_solved{ext}"

    updated_df = pd.DataFrame(updated_instances)
    updated_df.to_json(
        output_path,
        orient='records',
        lines=True,
        force_ascii=False,
    )



import json
from tqdm import tqdm
from src.models_new import MPMVSurrogate, MPMVOriginal
from src.utils import get_distribution_from_name
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
            u = inst['u']
            r = inst['r']
            v = inst['v']
            n_pick = inst['B']
            distr = get_distribution_from_name(inst['F'])
            C = tuple(inst['C'])

            sp = MPMVSurrogate(u, r, v, n_pick, distr, C, solver_params=solver_params)
            time_1 = time.time()
            sp_x = sp.solve(method='SP', n_steps=1001)
            time_2 = time.time()
            # rsp_x = sp.solve(method='RSP', n_steps=1001)
            time_3 = time.time()

            op = MPMVOriginal(u, r, v, n_pick, distr)
            random_comp = op.Generate_batch(n_samples=100000)
            op.set_random_comp(random_comp)


            pi_sp = op.Revenue(sp_x)
            # pi_rsp = op.Revenue(rsp_x)
            time_4 = time.time()

            inst["sp_x"] = np.array(sp_x).tolist()
            inst["time_sp_x"] = time_2 - time_1
            # inst["rsp_x"] = np.array(rsp_x).tolist()
            # inst["time_rsp_x"] = time_3 - time_2

            inst["pi_sp"] = pi_sp
            # inst["pi_rsp"] = pi_rsp

            inst["time_evaluate"] = time_4 - time_3

            updated_instances.append(inst)

    # with open(f"updated_{filename}", 'w', encoding='utf-8') as f:
    #     for inst in updated_instances:
    #         f.write(json.dumps(inst) + '\n')

    # updated_df = pd.DataFrame(updated_instances)
    # updated_df.to_json(
    #     f"updated_{filename}",
    #     orient='records',
    #     lines=True,
    #     force_ascii=False,
    # )

    root, ext = os.path.splitext(filename)
    output_path = f"{root}_solved{ext}"

    updated_df = pd.DataFrame(updated_instances)
    updated_df.to_json(
        output_path,
        orient='records',
        lines=True,
        force_ascii=False,
    )


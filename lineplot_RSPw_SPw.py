import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import MPMVSurrogate
from src.InstanceGenerator import InstanceGenerator
from src.utils import get_distribution_from_name


def compute_and_save(output_path, N_list, n_steps=301):
    """Compute RSP(w) and SP(w) curves and save to JSONL."""

    np.random.seed(0)
    instance_generator = InstanceGenerator()
    u_all, r_all, v_all = instance_generator.IND(N=100, N0=100)
    F = "GumBel"

    solver_params = {'Threads': 24}

    with open(output_path, 'w') as fout:
        for N in tqdm(N_list, desc="Computing RSP(w) vs SP(w)"):
            B           = round(0.1 * N)
            N0          = round(0.1 * N)
            C           = round(0.2 * N)
            cardinality = (0, C)

            u     = u_all[0:N]
            r     = r_all[0:N]
            v     = v_all[0:N0]
            distr = get_distribution_from_name(F)

            sp = MPMVSurrogate(u, r, v, B, distr, cardinality, solver_params=solver_params)
            low, high = sp._get_box_range()
            h_arr  = np.linspace(low, high, n_steps)
            LP_arr = np.zeros(n_steps)
            IP_arr = np.zeros(n_steps)

            for i, h in enumerate(h_arr):
                _, LP_arr[i] = sp.RSP(h)
                _, IP_arr[i] = sp.SP(h)

            record = {
                "N":      N,
                "h_arr":  h_arr.tolist(),
                "LP_arr": LP_arr.tolist(),
                "IP_arr": IP_arr.tolist(),
            }
            json.dump(record, fout)
            fout.write("\n")

    print(f"Data saved to {output_path}")


def load_and_plot(data_path, output_folder):
    """Load saved data and generate plots."""

    with open(data_path, 'r') as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            record = json.loads(s)

            N      = record["N"]
            h_arr  = np.array(record["h_arr"])
            LP_arr = np.array(record["LP_arr"])
            IP_arr = np.array(record["IP_arr"])

            plt.figure()
            plt.plot(h_arr, LP_arr, label="RSP(w)", color='red')
            plt.plot(h_arr, IP_arr, label="SP(w)",  color='blue')
            plt.xlabel("w")
            plt.ylabel("value")
            plt.legend()

            save_file = os.path.join(output_folder, f"lineplot_RSP(w)_SP(w)_N_{N}.pdf")
            plt.savefig(save_file, format='pdf')
            plt.close()
            print(f"Saved {save_file}")


if __name__ == '__main__':

    path      = "./data/RSPw2SPw/"
    data_file = os.path.join(path, "RSPw2SPw_data.jsonl")
    N_list    = [20, 30, 40, 50]

    os.makedirs(path, exist_ok=True)

    # Step 1: Compute and save (comment out if data already exists)
    compute_and_save(data_file, N_list)

    # Step 2: Load and plot
    load_and_plot(data_file, path)
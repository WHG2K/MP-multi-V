import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from src.models import MPMVSurrogate
from src.utils import get_distribution_from_name


def compute_and_save(data_path, output_path):
    """Solve SP for each instance and save results with sp_x."""

    with open(data_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue

            t_start = time.time()

            inst    = json.loads(s)
            N       = inst["N"]
            beta    = inst["beta"]
            eta     = inst["eta"]
            cor     = inst["cor"]
            u, r, v = inst["u"], inst["r"], inst["v"]

            B = round(beta * N)
            C = round(eta * N)

            sp   = MPMVSurrogate(u=u, r=r, v=v, n_pick=B,
                                 distr=get_distribution_from_name('GumBel'),
                                 C=(0, C))
            sp_x = sp.solve(method='SP', n_steps=1001)

            inst["sp_x"] = sp_x if isinstance(sp_x, list) else sp_x.tolist()
            json.dump(inst, fout)
            fout.write("\n")

            t_elapsed = round(time.time() - t_start, 1)
            print(f"N={N}, beta={beta}, eta={eta}, cor={cor} → {t_elapsed}s")

    print(f"Saved {output_path}")


def load_and_plot(data_path, output_folder):
    """Load solved data and generate scatter plots."""

    with open(data_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            inst = json.loads(s)
            N    = inst["N"]
            beta = inst["beta"]
            eta  = inst["eta"]
            cor  = inst["cor"]
            u, r = inst["u"], inst["r"]
            sp_x = inst["sp_x"]

            u_selected   = [ui for ui, xi in zip(u, sp_x) if xi > 0.9]
            r_selected   = [ri for ri, xi in zip(r, sp_x) if xi > 0.9]
            u_unselected = [ui for ui, xi in zip(u, sp_x) if xi < 0.1]
            r_unselected = [ri for ri, xi in zip(r, sp_x) if xi < 0.1]

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.scatter(u_unselected, r_unselected, s=80, color='red',   label='Not Offered', marker='o')
            ax.scatter(u_selected,   r_selected,   s=80, color='green', label='Offered',     marker='o')

            legend = ax.legend(frameon=True, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('grey')

            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            ax.set_xlabel('Base Utility', fontsize=14)
            ax.set_ylabel('Revenue', fontsize=14)

            plt.grid(False)
            plt.xticks(np.linspace(-2, 2, 5))
            plt.yticks(np.linspace(10, 100, 10))
            plt.xlim(-2, 2)
            plt.ylim(0, 100)

            save_file = os.path.join(
                output_folder,
                f"scatterplot_N-{N}_beta-{beta}_eta-{eta}_{cor}.pdf"
            )
            plt.savefig(save_file, format='pdf')
            plt.close()
            print(f"Saved {save_file}")


if __name__ == '__main__':

    data_file = "./data/heatmaps/HEATMAP_data.jsonl"
    base, ext = os.path.splitext(data_file)
    solved_file = base + "_solved" + ext

    # Step 1: Compute and save (comment out if data already exists)
    compute_and_save(data_file, solved_file)

    # Step 2: Load and plot
    load_and_plot(solved_file, os.path.dirname(data_file))
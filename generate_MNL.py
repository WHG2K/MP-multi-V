from src.InstanceGenerator import InstanceGenerator
# import pandas as 
import json
import os
import numpy as np

if __name__ == "__main__":

    np.random.seed(2025)
    # Experiment settings OLD
    # N_values = [20, 50, 100]
    # # N_values = [10,20]
    # K_values = [5]              # List of latent-class counts (can be expanded later)
    # eta_values = [1.0, 0.5, 0.2]
    # size_con_set_ratio = 1.0
    # instances_per_N = 10

    # Experiment settings
    N_values = [20, 50, 100, 200]
    C_values = [4, 6, 8]
    instances_per_N =  20
    # N_values = [20, 50]
    # C_values = [4, 6]
    # instances_per_N = 10
    gen = InstanceGenerator()

    # for method_name, cor_label in [('IND', 'ind'), ('LINEAR', 'linear')]:
    #     u, r, v = getattr(gen, method_name)(3, 1)
    #     print(u, r, v)
    #     break

    output_folder = "./data/MNL"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "MNL_data.jsonl")

    with open(output_file, "w") as fout:
            for N in N_values:
                for method_name, cor_label in [('IND', 'ind'), ('LINEAR', 'linear')]:
                    for _ in range(instances_per_N):
                        u, r, v = getattr(gen, method_name)(N, 1)
                        # pf, weights, r = gen.generate(r_range=(10, 100), size_C=N, sort=True)
                        pf = np.exp(np.array(u)-v[0]).tolist()
                        s = np.random.uniform(0, 2, size=N).tolist()
                        for C in C_values:
                            record = {
                                "N": N,
                                "C": C,
                                "cor": cor_label,
                                "r": r,
                                "pf": pf,
                                "s": s
                            }
                            json.dump(record, fout)
                            fout.write("\n")

    print(f"Created {output_file} with {instances_per_N * len(N_values) * len(C_values) * 2} instances.")
from src.InstanceGenerator import MixMNL_Generator
import numpy as np
import json
import os

if __name__ == "__main__":
    # Experiment settings
    # N_values = [20, 50, 100]
    N_values = [20, 50]
    K_values = [3]
    C_values = [4, 6, 8]
    size_con_set_ratio = 1.0
    instances_per_N = 5

    # Output folder for generated JSONL files
    output_dir = "./data/mixMNL"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "MIXMNL_data.jsonl")
    with open(output_file, "w") as fout:
        for K in K_values:
            for N in N_values:
                gen = MixMNL_Generator(N=N, K=K)
                for _ in range(instances_per_N):
                    pf, weights, r = gen.generate(r_range=(10, 100), size_C=int(round(size_con_set_ratio * N)), sort=True)
                    s = np.random.uniform(0, 2, size=N).tolist()
                    for c_ub in C_values:
                        C = (0, c_ub)
                        record = {
                            "N": N,
                            "K": K,
                            "C": C,
                            "W": c_ub,
                            "r": r,
                            "pf": pf,
                            "s": s,
                            "weights": weights
                        }
                        json.dump(record, fout)
                        fout.write("\n")

    print(f"Created {output_file} with {instances_per_N * len(N_values) * len(K_values) * len(C_values)} instances.")
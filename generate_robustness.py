from src.InstanceGenerator import InstanceGenerator
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    perturb_type = "BINARY"
    # perturb_type = "NORMAL"
    # perturb_type = "UNIFORM"
    params_generator = InstanceGenerator()
    B_list = [2]
    N_list = [25]
    F_list = ['GumBel', 'NegExp', 'NorMal', 'UniForm', 'BiNormal']
    C_list = [6]
    # sigma_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    sigma_list = [0, 0.1, 0.2, 0.3, 0.4]
    n_instances = 100  # number of instances to generate per method

    # Collect records across all distributions into a single file
    records = []
    for F in F_list:
        for B in B_list:
            for N in N_list:
                N0 = B
                # for method_name, cor_label in [('IND', 'ind'), ('LINEAR', 'linear')]:
                for method_name, cor_label in [('IND', 'ind')]:
                    for inst_id in range(1, n_instances + 1):
                        u, r, v = getattr(params_generator, method_name)(N, N0)
                        if perturb_type == "BINARY":
                            err = np.random.choice([-1, 1], size=N)
                            err2 = np.random.choice([-1, 1], size=N0)
                        elif perturb_type == "NORMAL":
                            err = np.random.randn(N)
                            err2 = np.random.randn(N0)
                        elif perturb_type == "UNIFORM":
                            err = np.random.uniform(-1, 1, N)
                            err2 = np.random.uniform(-1, 1, N0)
                        for c1 in C_list:
                            C = (0, c1)
                            for sigma in sigma_list:
                                # Mark as ground truth only when there is no perturbation
                                Is_GroundTruth = 1 if abs(sigma) < 1e-6 else 0

                                u_err = (np.array(u) + sigma * err).tolist()
                                v_err = (np.array(v) + sigma * err2).tolist()

                                records.append({
                                    'menu': inst_id,
                                    'Is_GroundTruth': Is_GroundTruth,
                                    'sigma': sigma,
                                    'N': N,
                                    'B': B,
                                    'C': C,
                                    'N0': N0,
                                    'F': F,
                                    'cor': cor_label,
                                    'u': u_err,
                                    'r': r,
                                    'v': v_err
                                })
                                

    # Build DataFrame
    df = pd.DataFrame.from_records(
        records,
        columns=['menu', 'Is_GroundTruth', 'sigma', 'N', 'B', 'C', 'N0', 'F', 'cor', 'u', 'r', 'v']
    )

    # Output folder for generated JSONL files
    output_dir = "./data/misspecification/"
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"ROBUST_data.jsonl"
    output_path = os.path.join(output_dir, file_name)
    df.to_json(output_path, orient='records', lines=True)
    print(f"Generated {len(df)} samples across all distributions, saved to {output_path}")
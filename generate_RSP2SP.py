from src.InstanceGenerator import InstanceGenerator
import pandas as pd
import os

if __name__ == "__main__":

    params_generator = InstanceGenerator()
    B_list = [2, 4]
    # N_list = [25, 50, 75, 100]
    N_list = [25, 50]
    F_list = ['GumBel', 'NegExp', 'NorMal', 'UniForm', 'BiNormal']
    eta = 0.2
    n_instances = 5  # Generate n_instances records for each method

    folder = "./data/RSP2SP/"
    os.makedirs(folder, exist_ok=True)

    # Collect records across all distributions into a single file
    records = []
    for F in F_list:
        for B in B_list:
            for N in N_list:
                N0 = B
                # Round eta*N and convert to integer
                C = [0, int(round(eta * N))]

                for method_name, cor_label in [('IND', 'ind'), ('LINEAR', 'linear')]:
                    for inst_id in range(1, n_instances + 1):
                        u, r, v = getattr(params_generator, method_name)(N, N0)
                        records.append({
                            'menu': inst_id,
                            'N': N,
                            'B': B,
                            'C': C,
                            'N0': N0,
                            'F': F,
                            'cor': cor_label,
                            'u': u,
                            'r': r,
                            'v': v
                        })

    # Build DataFrame and save all distributions into a single JSONL file
    df = pd.DataFrame.from_records(
        records,
        columns=['menu', 'N', 'B', 'C', 'N0', 'F', 'cor', 'u', 'r', 'v']
    )
    file_path = os.path.join(folder, "RSP2SP_data.jsonl")
    df.to_json(file_path, orient='records', lines=True)
    print(f"Generated {len(df)} samples across all distributions, saved to {file_path}")
from src.InstanceGenerator import InstanceGenerator
import pandas as pd
import os

if __name__ == "__main__":

    params_generator = InstanceGenerator()
    B_list = [2, 4]
    N_list = [100, 300, 1000, 3000, 10000]
    # N_list = [25, 50, 75, 100]
    # F_list = ['GumBel', 'NegExp', 'NorMal', 'UniForm', 'BiNormal']
    F_list = ['GumBel']
    # F_list = ['GumBel']
    # F_list = ['GumBel']
    eta = 0.1
    n_instances = 10  # Generate n_instances records for each method

    folder = "./data/RSP2SP/runtime/"
    os.makedirs(folder, exist_ok=True)

    # Save separately for each F
    for F in F_list:
        for B in B_list:
            records = []
            for N in N_list:
                N0 = B
                # Round eta*N and convert to integer
                C = [0, int(round(eta * N))]

                # for method_name, cor_label in [('IND', 'ind'), ('LINEAR', 'linear')]:
                for method_name, cor_label in [('IND', 'ind')]:
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

        # Build DataFrame and save as a JSONL file for the corresponding F
        df = pd.DataFrame.from_records(
            records,
            columns=['menu', 'N', 'B', 'C', 'N0', 'F', 'cor', 'u', 'r', 'v']
        )
        file_name = f"RSP2SP_RUNTIME_data_eta_{eta}.jsonl"
        file_path = folder + file_name
        df.to_json(file_path, orient='records', lines=True)
        print(f"Generated {len(df)} samples for '{F}', saved to {file_path}")
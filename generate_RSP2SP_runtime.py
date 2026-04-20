import os
import pandas as pd
from src.InstanceGenerator import InstanceGenerator

if __name__ == "__main__":

    params_generator = InstanceGenerator()

    # === Configuration ===
    B_list      = [2, 4]
    # N_list      = [100, 300, 1000, 3000, 10000]
    N_list      = [100, 300, 1000, 2000]
    F           = 'GumBel'
    eta         = 0.2
    n_instances = 3

    folder = "./data/RSP2SP/"
    os.makedirs(folder, exist_ok=True)

    records = []
    for B in B_list:
        for N in N_list:
            N0 = B
            C  = [0, int(round(eta * N))]

            for method_name, cor_label in [('IND', 'ind')]:
                for inst_id in range(1, n_instances + 1):
                    u, r, v = getattr(params_generator, method_name)(N, N0)
                    records.append({
                        'menu': inst_id,
                        'N':    N,
                        'B':    B,
                        'C':    C,
                        'N0':   N0,
                        'F':    F,
                        'cor':  cor_label,
                        'u':    u,
                        'r':    r,
                        'v':    v
                    })

    # Build DataFrame and save as a JSONL file
    df = pd.DataFrame.from_records(
        records,
        columns=['menu', 'N', 'B', 'C', 'N0', 'F', 'cor', 'u', 'r', 'v']
    )
    file_name = f"RSP2SP_RUNTIME_data.jsonl"
    file_path = os.path.join(folder, file_name)
    df.to_json(file_path, orient='records', lines=True)
    print(f"Generated {len(df)} samples for '{F}', saved to {file_path}")
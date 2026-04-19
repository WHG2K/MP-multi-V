from src.InstanceGenerator import InstanceGenerator
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    params_generator = InstanceGenerator()

    # ---------- configuration ----------
    F_list = ['GumBel', 'NegExp', 'UniForm', 'NorMal', 'BiNormal']
    # F_list = ['GumBel', 'NegExp']
    cor_list = ['ind', 'linear']
    n_instances = 20 # menu ids 1..10

    # Max N and N0 — generate once at these sizes, then slice for smaller combos
    N_max = 25
    N0_max = 4

    # All (N, B, C, N0) combos to emit
    param_combos = [
        (15, 2, [0, 6], 4),
        (15, 4, [0, 6], 4),
        (25, 2, [0, 6], 4),
    ]

    # ---------- pre-generate (cor, menu) -> (u, r, v) ----------
    data_cache = {}
    for cor in cor_list:
        for menu_id in range(1, n_instances + 1):
            if cor == 'ind':
                u_full, r_full, v_full = params_generator.IND(N_max, N0_max)
            elif cor == 'linear':
                u_full, r_full, v_full = params_generator.LINEAR(N_max, N0_max)
            data_cache[(cor, menu_id)] = (u_full, r_full, v_full)

    # ---------- emit rows: F -> param_combo -> cor -> menu ----------
    records = []
    for F in F_list:
        for (N, B, C, N0) in param_combos:
            for cor in cor_list:
                for menu_id in range(1, n_instances + 1):
                    u_full, r_full, v_full = data_cache[(cor, menu_id)]
                    records.append({
                        'menu': menu_id,
                        'N':    N,
                        'B':    B,
                        'C':    C,
                        'F':    F,
                        'cor':  cor,
                        'u':    u_full[:N],
                        'r':    r_full[:N],
                        'v':    v_full[:N0],
                        'N0':   N0,
                    })

    # ---------- save ----------
    save_dir = "./data/SP2OP/"
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame.from_records(
        records,
        columns=['menu', 'N', 'B', 'C', 'F', 'cor', 'u', 'r', 'v', 'N0']
    )
    file_name = os.path.join(save_dir, "SP2OP_data.jsonl")
    df.to_json(file_name, orient='records', lines=True)
    print(f"Generated {len(df)} rows, saved to {file_name}")

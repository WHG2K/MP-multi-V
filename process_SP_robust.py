import os
import pandas as pd
from tqdm import tqdm
from src.models import MPMVOriginal
from src.utils import get_distribution_from_name


if __name__ == "__main__":

    folder    = "./data/misspecification/"
    # folder    = "./paper data/"
    file_name = "ROBUST_data_solved.jsonl"
    out_name  = "ROBUST_data_processed.jsonl"

    df = pd.read_json(os.path.join(folder, file_name), lines=True)

    n_rows = 5  # 1 ground-truth (sigma=0) + 4 perturbed sigma levels
    df["pi_sp_err"]       = pd.NA
    df["gap_sp_err_2_sp"] = pd.NA

    for start in tqdm(range(0, len(df), n_rows), desc="Processing"):
        group = df.iloc[start : start + n_rows]
        if group.empty:
            break

        row0   = group.iloc[0]
        distr0 = get_distribution_from_name(row0["F"])
        op     = MPMVOriginal(row0["u"], row0["r"], row0["v"], row0["B"], distr0)

        random_comp = op.Generate_batch(n_samples=100_000)
        op.set_random_comp(random_comp)

        # Ground-truth revenue (sigma=0)
        pi0 = op.Revenue(row0["sp_x"])
        df.at[group.index[0], "pi_sp_err"]       = pi0
        df.at[group.index[0], "gap_sp_err_2_sp"] = 0.0

        # Perturbed cases
        for idx in group.index[1:]:
            pi_i = op.Revenue(df.at[idx, "sp_x"])
            df.at[idx, "pi_sp_err"]       = pi_i
            df.at[idx, "gap_sp_err_2_sp"] = 1 - (pi_i / pi0)

    out_path = os.path.join(folder, out_name)
    df.to_json(out_path, orient="records", lines=True)
    print(f"Processed {len(df)} rows → {out_path}")
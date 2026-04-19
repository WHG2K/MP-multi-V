import pandas as pd
import numpy as np
from src.models_new import MPMVOriginal
from src.utils import get_distribution_from_name
from tqdm import tqdm


# ── STEP 1: Process ──────────────────────────────────────────────────────────

def process(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """Add pi_sp_err and gap_sp_err_2_sp columns to df."""
    df = df.copy()
    df["pi_sp_err"]       = pd.NA
    df["gap_sp_err_2_sp"] = pd.NA

    for start in tqdm(range(0, len(df), n_rows)):
        group = df.iloc[start : start + n_rows]
        if group.empty:
            break

        row0   = group.iloc[0]
        distr0 = get_distribution_from_name(row0["F"])
        op     = MPMVOriginal(row0["u"], row0["r"], row0["v"], row0["B"], distr0)

        random_comp = op.Generate_batch(n_samples=100_000)
        op.set_random_comp(random_comp)

        # First row in the group is the ground-truth denominator
        pi0 = op.Revenue(row0["sp_x"])
        df.at[group.index[0], "pi_sp_err"]       = pi0
        df.at[group.index[0], "gap_sp_err_2_sp"] = 0.0

        for idx in group.index[1:]:
            pi_i = op.Revenue(df.at[idx, "sp_x"])
            df.at[idx, "pi_sp_err"]       = pi_i
            df.at[idx, "gap_sp_err_2_sp"] = 1 - (pi_i / pi0)

    return df


# ── STEP 2: LaTeX table ───────────────────────────────────────────────────────

def build_latex_table(df: pd.DataFrame) -> str:
    chosen_stats = ["mean", "p95"]

    F_map = {
        "GumBel":   "Gumbel",
        "NegExp":   "NegExp",
        "NorMal":   "Normal",
        "UniForm":  "Uniform",
        "BiNormal": "BiNormal",
    }
    df = df.copy()
    df["F"] = df["F"].map(F_map)

    alphas    = [0.1, 0.2, 0.3, 0.4]
    ordered_F = ["Gumbel", "NegExp", "Normal", "Uniform", "BiNormal"]

    stat_labels = {
        "mean": "Mean Gap",
        "p95":  "95\\% Gap",
        "max":  "Max Gap",
    }

    stats = {f: {s: [] for s in chosen_stats} for f in ordered_F}

    for f in ordered_F:
        for a in alphas:
            vals = (
                df[(df["F"] == f) & (np.isclose(df["sigma"], a))]
                ["gap_sp_err_2_sp"]
                .dropna() * 100
            )
            if "mean" in chosen_stats:
                stats[f]["mean"].append(vals.mean() if len(vals) else np.nan)
            if "p95" in chosen_stats:
                stats[f]["p95"].append(np.percentile(vals, 95) if len(vals) else np.nan)
            if "max" in chosen_stats:
                stats[f]["max"].append(vals.max() if len(vals) else np.nan)

    n_groups = len(chosen_stats)
    col_fmt  = "l" + ("c" + "r" * len(alphas)) * n_groups
    # drop trailing separator added by the pattern
    col_fmt  = "l" + "c" + ("r" * len(alphas) + "c") * n_groups
    col_fmt  = col_fmt.rstrip("c")

    groups = [stat_labels[s] for s in chosen_stats]
    alpha_block = " & ".join(r"$\alpha=%.1f$" % a for a in alphas)

    # cmidrule positions
    cmid_rules, start = [], 3
    for _ in groups:
        end = start + len(alphas) - 1
        cmid_rules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 2

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{" + col_fmt + "}",
        r"    \toprule",
        "    & & " +
            " & & ".join(
                r"\multicolumn{%d}{c}{%s ($\alpha$)}" % (len(alphas), g)
                for g in groups
            ) + r" \\",
        "    & & " +
            " & & ".join(alpha_block for _ in groups) + r" \\",
        "    " + " ".join(cmid_rules),
    ]

    for f in ordered_F:
        raw = [f]
        for stat in chosen_stats:
            raw += [f"{v:.2f}\\%" for v in stats[f][stat]]
            raw += [""]          # column separator
        raw = raw[:-1]           # drop trailing separator
        lines.append("    " + " & ".join(raw) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Suboptimality gap statistics by distribution and $\alpha$, shown as percentages.}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    folder    = "./data/misspecification/"
    file_name = "ROBUST_data_solved.jsonl"

    # 1. Load
    df = pd.read_json(folder + file_name, lines=True)

    # 2. Process  (n_rows=5: 1 ground-truth + 4 perturbed sigma levels)
    df = process(df, n_rows=5)

    # 3. Save processed file
    out_path = folder + "Processed_" + file_name
    df.to_json(out_path, orient="records", lines=True)
    print(f"Processed {len(df)} rows → {out_path}\n")

    # 4. Print LaTeX table
    print(build_latex_table(df))
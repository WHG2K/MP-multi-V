import os
import numpy as np
import pandas as pd


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
    stat_labels = {"mean": "Mean Gap", "p95": "95\\% Gap"}

    stats = {f: {s: [] for s in chosen_stats} for f in ordered_F}
    for f in ordered_F:
        for a in alphas:
            vals = (
                df[(df["F"] == f) & (np.isclose(df["sigma"], a))]
                ["gap_sp_err_2_sp"].dropna() * 100
            )
            if "mean" in chosen_stats:
                stats[f]["mean"].append(vals.mean() if len(vals) else np.nan)
            if "p95" in chosen_stats:
                stats[f]["p95"].append(np.percentile(vals, 95) if len(vals) else np.nan)

    n_groups    = len(chosen_stats)
    col_fmt     = "l" + "c" + ("r" * len(alphas) + "c") * n_groups
    col_fmt     = col_fmt.rstrip("c")
    groups      = [stat_labels[s] for s in chosen_stats]
    alpha_block = " & ".join(r"$\alpha=%.1f$" % a for a in alphas)

    cmid_rules, start_col = [], 3
    for _ in groups:
        end_col = start_col + len(alphas) - 1
        cmid_rules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 2

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{" + col_fmt + "}",
        r"    \toprule",
        "    & & " + " & & ".join(
            r"\multicolumn{%d}{c}{%s ($\alpha$)}" % (len(alphas), g) for g in groups
        ) + r" \\",
        "    & & " + " & & ".join(alpha_block for _ in groups) + r" \\",
        "    " + " ".join(cmid_rules),
    ]
    for f in ordered_F:
        raw = [f]
        for stat in chosen_stats:
            raw += [f"{v:.2f}\\%" for v in stats[f][stat]]
            raw += [""]
        raw = raw[:-1]
        lines.append("    " + " & ".join(raw) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Suboptimality gap statistics by distribution and $\alpha$, shown as percentages.}",
        r"\end{table}",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    jsonl_path = "./data/misspecification/ROBUST_data_processed.jsonl"

    df = pd.read_json(jsonl_path, lines=True)
    print(build_latex_table(df))
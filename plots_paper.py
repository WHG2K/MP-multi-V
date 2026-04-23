import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# === Configuration ===
DATA_FOLDER   = "./paper data"
OUTPUT_FOLDER = "./paper data/outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_fig(filename, **kwargs):
    """Save figure to OUTPUT_FOLDER, close it, and print the path."""
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, **kwargs)
    plt.close()
    print(f"Saved {path}")


def plot_scatterplots():
    """Figures 1 and L1: SP-based Assortment Scatter Plots"""

    df_path = os.path.join(DATA_FOLDER, "HEATMAP_data_solved.jsonl")

    figure_map = {
        (20,  0.2, "ind"): "figure_1a.pdf",
        (20,  0.4, "ind"): "figure_1b.pdf",
        (50,  0.2, "ind"): "figure_1c.pdf",
        (50,  0.4, "ind"): "figure_1d.pdf",
        (100, 0.2, "ind"): "figure_1e.pdf",
        (100, 0.4, "ind"): "figure_1f.pdf",
        (200, 0.2, "ind"): "figure_1g.pdf",
        (200, 0.4, "ind"): "figure_1h.pdf",
        (20,  0.2, "linear"): "figure_L1a.pdf",
        (20,  0.4, "linear"): "figure_L1b.pdf",
        (50,  0.2, "linear"): "figure_L1c.pdf",
        (50,  0.4, "linear"): "figure_L1d.pdf",
        (100, 0.2, "linear"): "figure_L1e.pdf",
        (100, 0.4, "linear"): "figure_L1f.pdf",
        (200, 0.2, "linear"): "figure_L1g.pdf",
        (200, 0.4, "linear"): "figure_L1h.pdf",
    }

    with open(df_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            inst = json.loads(s)
            N    = inst["N"]
            eta  = inst["eta"]
            cor  = inst["cor"]
            u, r = inst["u"], inst["r"]
            sp_x = inst["sp_x"]

            key = (N, eta, cor)
            if key not in figure_map:
                continue

            u_selected   = [ui for ui, xi in zip(u, sp_x) if xi > 0.9]
            r_selected   = [ri for ri, xi in zip(r, sp_x) if xi > 0.9]
            u_unselected = [ui for ui, xi in zip(u, sp_x) if xi < 0.1]
            r_unselected = [ri for ri, xi in zip(r, sp_x) if xi < 0.1]

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.scatter(u_unselected, r_unselected, s=80, color='red',   label='Not Offered', marker='o')
            ax.scatter(u_selected,   r_selected,   s=80, color='green', label='Offered',     marker='o')

            legend = ax.legend(frameon=True, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('grey')

            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            ax.set_xlabel('Base Utility', fontsize=14)
            ax.set_ylabel('Revenue', fontsize=14)

            plt.grid(False)
            plt.xticks(np.linspace(-2, 2, 5))
            plt.yticks(np.linspace(10, 100, 10))
            plt.xlim(-2, 2)
            plt.ylim(0, 100)

            save_fig(figure_map[key], format='pdf')


def plot_SP2OP():
    """Figures 2 and L2: SP vs. Optimal Assortment (Multi-purchase setting)"""

    df = pd.read_json(os.path.join(DATA_FOLDER, "SP2OP_data_solved.jsonl"), lines=True)

    def assign_p_type(row):
        if row["N"] == 15 and row["B"] == 2:
            return "$N=15$\n$B=2$"
        elif row["N"] == 25:
            return "$N$ \u2191 25\n$B=2$"
        elif row["B"] == 4:
            return "$N=15$\n$B$ \u2191 4"
        else:
            raise ValueError(f"Unknown configuration: N={row['N']}, B={row['B']}")

    df["p_type"]    = df.apply(assign_p_type, axis=1)
    df["RelGap_sp"] = (df["pi_bf"] - df["pi_sp"]) / df["pi_bf"]

    orders = [
        '$N=15$\n$B=2$',
        '$N$ ↑ 25\n$B=2$',
        '$N=15$\n$B$ ↑ 4'
    ]

    figure_map = {"ind": "figure_2.pdf", "linear": "figure_L2.pdf"}

    for cor_type, filename in figure_map.items():
        df_sub = df[df["cor"] == cor_type].copy()

        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_sub,
            x="p_type",
            y="RelGap_sp",
            hue="F",
            order=orders,
            showfliers=False
        )

        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel('RelGap')
        ax.legend(title="")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        save_fig(filename, format="pdf")


def plot_RSP2SP():
    """Figures 3 and L3: RSP vs. SP Solution Quality"""

    df = pd.read_json(os.path.join(DATA_FOLDER, "RSP2SP_data_solved.jsonl"), lines=True)

    df["RelGap_rsp2sp"] = (df["pi_sp"] - df["pi_rsp"]) / df["pi_sp"]

    F_mapping = {
        "GumBel":   "Gumbel",
        "NegExp":   "NegExp",
        "UniForm":  "Uniform",
        "NorMal":   "Normal",
        "BiNormal": "MixNormal"
    }
    df["F"] = df["F"].map(F_mapping)

    figure_map = {
        ("ind", 2):    "figure_3a.pdf",
        ("ind", 4):    "figure_3b.pdf",
        ("linear", 2): "figure_L3a.pdf",
        ("linear", 4): "figure_L3b.pdf",
    }

    for (cor_val, B_val), filename in figure_map.items():
        df_sub = df[(df['cor'] == cor_val) & (df['B'] == B_val)].copy()

        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_sub,
            x="N",
            y="RelGap_rsp2sp",
            hue="F",
            showfliers=False
        )

        ax.set_title("")
        ax.set_xlabel("N")
        ax.set_ylabel('RelGap')
        ax.legend(title="")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        if cor_val == "ind":
            low, high = -0.02, 0.10
        else:
            low, high = -0.04, 0.10

        pad = (high - low) * 0.05
        plt.ylim(low - pad, high + pad)

        save_fig(filename, format="pdf", bbox_inches="tight")


def plot_RSP2SP_runtime():
    """Figure 4: SP vs. RSP Runtime Comparison"""

    df = pd.read_json(os.path.join(DATA_FOLDER, "RSP2SP_RUNTIME_data_solved.jsonl"), lines=True)

    agg_df = df.groupby(['B', 'N'])[['time_sp_x', 'time_rsp_x']].mean().reset_index()
    agg_df = agg_df.rename(columns={'time_sp_x': 'SP', 'time_rsp_x': 'RSP'})

    melted_df = agg_df.melt(
        id_vars=['B', 'N'],
        value_vars=['SP', 'RSP'],
        var_name='Method',
        value_name='AvgTime'
    )

    figure_map = {2: "figure_4a.pdf", 4: "figure_4b.pdf"}

    for b, filename in figure_map.items():
        plt.figure(figsize=(8, 5))
        subset = melted_df[melted_df['B'] == b]
        sns.lineplot(data=subset, x='N', y='AvgTime', hue='Method', marker='o')
        plt.xlabel("N")
        plt.ylabel("seconds")
        plt.legend(title=None)

        save_fig(filename, dpi=300, bbox_inches='tight')


def plot_MNL():
    """Figures 5 and L4: SP vs. Optimal under Single-Purchase MNL"""

    C_list    = [4, 6, 8]
    hue_order = [f"$C={c:g}$" for c in C_list]
    palette   = ['#4C72B0', '#55A868', '#DD8452']

    figure_map = {
        ("cardinality", "ind"):    "figure_5a.pdf",
        ("cardinality", "linear"): "figure_L4a.pdf",
        ("spaceconstr", "ind"):    "figure_5b.pdf",
        ("spaceconstr", "linear"): "figure_L4b.pdf",
    }

    for constraint_type in ["cardinality", "spaceconstr"]:
        jsonl_path = os.path.join(DATA_FOLDER, f"MNL_data_{constraint_type}_solved.jsonl")

        records = []
        with open(jsonl_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                s = line.strip()
                if not s:
                    continue
                try:
                    records.append(json.loads(s))
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(records)
        df['opt_gap'] = 1 - df['pi_sp'] / df['pi_milp']
        df['C_str']   = df['C'].apply(lambda c: f"$C={c}$")
        df['C_str']   = pd.Categorical(df['C_str'], categories=hue_order, ordered=True)

        for cor_name in ["ind", "linear"]:
            filename = figure_map[(constraint_type, cor_name)]
            df_cor   = df[df["cor"] == cor_name]

            fig, ax = plt.subplots()
            sns.boxplot(
                data=df_cor,
                x="N",
                y="opt_gap",
                hue="C_str",
                hue_order=hue_order,
                showfliers=False,
                palette=palette
            )

            ax.set_xlabel("N")
            ax.set_ylabel("OptGap")
            ax.legend(title="")

            if cor_name == "ind":
                low, high = 0, 0.004
            else:
                low, high = 0, 0.016
            pad = (high - low) * 0.05
            ax.set_ylim(low - pad, high + pad)

            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

            save_fig(filename, format="pdf", dpi=300)


def plot_mixMNL():
    """Figure 6: SP vs. Optimal under Mixed MNL"""

    C_list    = [4, 6, 8]
    hue_order = [f"$C={c:g}$" for c in C_list]
    palette   = ['#4C72B0', '#55A868', '#DD8452']

    figure_map = {
        "cardinality": "figure_6a.pdf",
        "spaceconstr": "figure_6b.pdf",
    }

    for constraint_type, filename in figure_map.items():
        jsonl_path = os.path.join(DATA_FOLDER, f"MIXMNL_data_{constraint_type}_solved.jsonl")

        records = []
        with open(jsonl_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                s = line.strip()
                if not s:
                    continue
                try:
                    records.append(json.loads(s))
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(records)
        df['C1']      = df['C'].apply(lambda c: c[1])
        df['opt_gap'] = 1 - df['pi_baye'] / df['pi_milp']
        df['C_str']   = df['C1'].apply(lambda c: f"$C={c}$")
        df['C_str']   = pd.Categorical(df['C_str'], categories=hue_order, ordered=True)

        fig, ax = plt.subplots()
        sns.boxplot(
            data=df,
            x="N",
            y="opt_gap",
            hue="C_str",
            hue_order=hue_order,
            showfliers=False,
            palette=palette
        )

        ax.set_xlabel("N")
        ax.set_ylabel("OptGap")
        ax.legend(title="")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        low, high = 0.0, 0.05
        pad       = (high - low) * 0.05
        plt.ylim(low - pad, high + pad)

        save_fig(filename, format="pdf", dpi=300)


def table_SP_robust():
    """Table 1: Robustness to Misspecification — saves LaTeX table as .txt"""

    df = pd.read_json(os.path.join(DATA_FOLDER, "ROBUST_data_processed.jsonl"), lines=True)

    chosen_stats = ["mean", "p95"]
    F_map = {
        "GumBel":   "Gumbel",
        "NegExp":   "NegExp",
        "NorMal":   "Normal",
        "UniForm":  "Uniform",
        "BiNormal": "BiNormal",
    }
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

    latex_str = "\n".join(lines)
    out_path  = os.path.join(OUTPUT_FOLDER, "table_1.txt")
    with open(out_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved {out_path}")


def plot_RSPw_SPw():
    """Figure 7: RSP(w) vs. SP(w) Value Curves"""

    data_path  = os.path.join(DATA_FOLDER, "RSPw2SPw_data.jsonl")
    figure_map = {20: "figure_7a.pdf", 30: "figure_7b.pdf",
                  40: "figure_7c.pdf", 50: "figure_7d.pdf"}

    with open(data_path, 'r') as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            record = json.loads(s)

            N      = record["N"]
            h_arr  = np.array(record["h_arr"])
            LP_arr = np.array(record["LP_arr"])
            IP_arr = np.array(record["IP_arr"])

            plt.figure()
            plt.plot(h_arr, LP_arr, label="RSP(w)", color='red')
            plt.plot(h_arr, IP_arr, label="SP(w)",  color='blue')
            plt.xlabel("w")
            plt.ylabel("value")
            plt.legend()

            save_fig(figure_map[N], format='pdf')


if __name__ == "__main__":
    plot_scatterplots()
    plot_SP2OP()
    plot_RSP2SP()
    plot_RSP2SP_runtime()
    plot_MNL()
    plot_mixMNL()
    table_SP_robust()
    plot_RSPw_SPw()
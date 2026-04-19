import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick


###### SPLIT IND AND LINEAR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# === User configuration ===
# jsonl_path     = "data/mixedMNL/K5/consideration_set_size_0.3/ALL_mixmnl_con_set_0.3.jsonl"       # Input JSONL file path
# jsonl_path     = "./data/mixedMNL/Find_case_RO_really_bad/BO_RO_SPAO_MIXMNL_SMALLER_K_3_r_range_10_100_sorted.jsonl"       # Input JSONL file path
# results_folder = "./data/mixedMNL/Find_case_RO_really_bad"                    # Output chart directory
jsonl_path     = "./data/MNL/MNL_data_spaceconstr_solved.jsonl"       # Input JSONL file path
results_folder = "./data/MNL"                    # Output chart directory
C_list       = [4, 6, 8]              # List of C values to plot
# hue_order = [f"$\\eta={e:.2f}$" for e in eta_list]

# === Ensure output directory exists ===
os.makedirs(results_folder, exist_ok=True)

# === Manually read and parse JSONL, skipping empty lines ===
records = []
with open(jsonl_path, 'r', encoding='utf-8') as fin:
    for line in fin:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            records.append(obj)
        except json.JSONDecodeError:
            # Skip invalid lines
            continue

df = pd.DataFrame(records)
# df['eta'] = df['C']
# df['eta'] = df.apply(lambda row: row['C'][1] / row['N'], axis=1)
# df = df[df['eta'].isin(eta_list)]
df['C1'] = df['C']
df['opt_gap'] = 1 - df['pi_sp'] / df['pi_milp']



# === Prepare plotting fields ===
df['C_str'] = df['C1'].apply(lambda c: f"$C={c}$")
hue_order      = [f"$C={e:g}$" for e in C_list]

df['C_str'] = pd.Categorical(df['C_str'], categories=hue_order, ordered=True)

# === Color palette ===
palette = [
    '#4C72B0', '#55A868', '#DD8452',
    '#C44E52', '#8172B3', '#937860',
    '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD'
]
pal = palette[:len(C_list)]

for cor_name in ["ind", "linear"]:

    df_cor = df[df["cor"] == cor_name]

    # === Draw boxplot ===
    fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df_cor,
        x="N",
        y="opt_gap",
        hue="C_str",
        hue_order=hue_order,
        showfliers=False,
        palette=pal
    )

    # === Axis labels and formatting ===
    ax.set_xlabel("N")
    ax.set_ylabel("OptGap")

    if cor_name == "ind":
        low, high = 0, 0.004
    else:
        low, high = 0, 0.016
    pad_ratio = 0.05  
    pad = (high - low) * pad_ratio
    ax.set_ylim(low - pad, high + pad)

    # ymin, ymax = df['opt_gap'].min(), df['opt_gap'].max()
    # margin = (ymax - ymin) * 0.1
    # plt.ylim(ymin - margin, ymax + margin)

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))
    # plt.yticks([0, 0.002, 0.004, 0.006, 0.008, 0.01])

    ax.legend(title="")
    # plt.title("Surrogate SP vs Mixed-MNL Optimality Gap")

    # === Save chart ===
    C_str = "-".join(str(e) for e in C_list)
    out_file = os.path.join(
        results_folder,
        f"MNL_spaceconstr_boxplot_cor_{cor_name}"
    )
    # plt.savefig(out_file, format="pdf", dpi=300, bbox_inches='tight')
    plt.savefig(out_file + ".pdf", format="pdf", dpi=300)
    # plt.savefig(out_file + ".jpg", format="jpg", dpi=300)
    plt.close()

    print(f"Boxplot saved to {out_file}")
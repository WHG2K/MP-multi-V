import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick


###### SPLIT IND AND LINEAR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# === User configuration ===
jsonl_path = "./data/MNL/MNL_data_cardinality_solved.jsonl"  # Input JSONL file path
C_list     = [4, 6, 8]                                       # List of C values to plot

# === Derive output folder from input file path ===
results_folder = os.path.dirname(jsonl_path)

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
            continue

df = pd.DataFrame(records)
df['C1']      = df['C']
df['opt_gap'] = 1 - df['pi_sp'] / df['pi_milp']

# === Prepare plotting fields ===
df['C_str'] = df['C1'].apply(lambda c: f"$C={c}$")
hue_order   = [f"$C={e:g}$" for e in C_list]
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
    pad       = (high - low) * pad_ratio
    ax.set_ylim(low - pad, high + pad)

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

    ax.legend(title="")

    # === Save chart ===
    out_file = os.path.join(
        results_folder,
        f"boxplot_MNL_cardinality_{cor_name}"
    )
    plt.savefig(out_file + ".pdf", format="pdf", dpi=300)
    plt.close()

    print(f"Boxplot saved to {out_file}.pdf")
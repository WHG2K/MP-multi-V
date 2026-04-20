import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# === Configuration ===
jsonl_path = "./data/mixMNL/MIXMNL_data_spaceconstr_solved.jsonl"
C_list     = [4, 6, 8]

# === Derive output folder from input file path ===
results_folder = os.path.dirname(jsonl_path)

# === Manually read and parse JSONL, skipping blank lines ===
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
df['C1']      = df['C'].apply(lambda c: c[1])
df['opt_gap'] = 1 - df['pi_baye'] / df['pi_milp']

# === Prepare plot fields ===
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

# === Draw boxplot ===
fig, ax = plt.subplots()
sns.boxplot(
    data=df,
    x="N",
    y="opt_gap",
    hue="C_str",
    hue_order=hue_order,
    showfliers=False,
    palette=pal
)

# === Axis formatting ===
ax.set_xlabel("N")
ax.set_ylabel("OptGap")
ax.legend(title="")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

low, high = 0.0, 0.05
pad       = (high - low) * 0.05
plt.ylim(low - pad, high + pad)

# === Save figure ===
out_file = os.path.join(results_folder, "boxplot_mixmnl_spaceconstr")
plt.savefig(out_file + ".pdf", format="pdf", dpi=300)
plt.close()

print(f"Boxplot saved to {out_file}.pdf")
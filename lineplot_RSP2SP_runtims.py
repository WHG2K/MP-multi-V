import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
jsonl_path = "./data/RSP2SP/RSP2SP_RUNTIME_data_solved.jsonl"

# === Derive output folder from input file path ===
folder = os.path.dirname(jsonl_path)

# === Load jsonl data ===
df = pd.read_json(jsonl_path, lines=True)

# Group and compute average values
agg_df = df.groupby(['B', 'N'])[['time_sp_x', 'time_rsp_x']].mean().reset_index()

# Rename columns before melting
agg_df = agg_df.rename(columns={'time_sp_x': 'SP', 'time_rsp_x': 'RSP'})

# Melt the dataframe for seaborn plotting
melted_df = agg_df.melt(
    id_vars=['B', 'N'],
    value_vars=['SP', 'RSP'],
    var_name='Method',
    value_name='AvgTime'
)

# Plot for each B
Bs = sorted(melted_df['B'].unique())

for b in Bs:
    plt.figure(figsize=(8, 5))
    subset = melted_df[melted_df['B'] == b]
    sns.lineplot(data=subset, x='N', y='AvgTime', hue='Method', marker='o')
    plt.xlabel("N")
    plt.ylabel("seconds")
    plt.legend(title=None)

    out_file = os.path.join(folder, f"plot_runtims_rsp_sp_B_{b}.pdf")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
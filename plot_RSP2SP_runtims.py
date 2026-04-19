import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load jsonl data
folder = "./data/RSP2SP/runtime/"
df = pd.read_json(folder + "RSP2SP_RUNTIME_data_eta_0.1_solved.jsonl", lines=True)

# Group and compute average values
agg_df = df.groupby(['B', 'N'])[['time_sp_x', 'time_rsp_x']].mean().reset_index()

# Rename columns before melting
agg_df = agg_df.rename(columns={'time_sp_x': 'SP', 'time_rsp_x': 'RSP'})

# Melt the dataframe for seaborn plotting
melted_df = agg_df.melt(id_vars=['B', 'N'], 
                        value_vars=['SP', 'RSP'],
                        var_name='Method',
                        value_name='AvgTime')

# Plot for each B
Bs = sorted(melted_df['B'].unique())

for b in Bs:
    plt.figure(figsize=(8, 5))
    subset = melted_df[melted_df['B'] == b]
    # print(f"B = {b}")
    # print("sp average time: ", subset[subset['Method'] == 'SP']['AvgTime'].mean())
    # print("rsp average time: ", subset[subset['Method'] == 'RSP']['AvgTime'].mean())
    # print("--------------------------------")
    sns.lineplot(data=subset, x='N', y='AvgTime', hue='Method', marker='o')
    plt.xlabel("N")
    plt.ylabel("seconds")
    # plt.grid(True)
    plt.legend(title=None)  # remove legend title
    plt.savefig(folder + f"plot_runtims_rsp_sp_B_{b}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
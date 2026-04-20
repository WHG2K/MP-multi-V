import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if __name__ == "__main__":

    # === Configuration ===
    jsonl_path = "./data/RSP2SP/RSP2SP_data_solved.jsonl"

    # === Derive output folder from input file path ===
    folder_path = os.path.dirname(jsonl_path)

    # === Read data ===
    df = pd.read_json(jsonl_path, lines=True)

    # Column needed for the boxplot
    df["RelGap_rsp2sp"] = (df["pi_sp"] - df["pi_rsp"]) / df["pi_sp"]

    F_list    = ["GumBel", "NegExp", "UniForm", "NorMal", "BiNormal"]
    F_legends = ["Gumbel", "NegExp", "Uniform", "Normal", "MixNormal"]

    F_mapping = dict(zip(F_list, F_legends))
    df["F"]   = df["F"].map(F_mapping)

    # Group and plot
    for cor_val in ["ind", "linear"]:

        for B_val in [2, 4]:

            df_sub = df[(df['cor'] == cor_val) & (df['B'] == B_val)].copy()

            ####################################################
            ############### SP suboptimality gap ###############
            ####################################################

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

            pad_ratio = 0.05
            pad       = (high - low) * pad_ratio
            plt.ylim(low - pad, high + pad)

            out_file = os.path.join(folder_path, f"boxplot_RSP2SP_B_{B_val}_{cor_val}.pdf")
            plt.savefig(out_file, format="pdf", bbox_inches="tight")
            plt.close()